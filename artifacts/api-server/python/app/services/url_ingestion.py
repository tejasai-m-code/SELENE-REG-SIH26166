"""URL image ingestion service with strict SSRF protection and Google Drive support.

Isolated input-handling module designed to fetch public HTTP/HTTPS images and
public Google Drive links, validate image integrity using the project's OpenCV stack,
and provide safe local artifacts for the registration pipeline.
"""

from __future__ import annotations

import hashlib
import ipaddress
import mimetypes
import re
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import cv2
import numpy as np
import requests

from app.utils.image_utils import ALLOWED_EXTENSIONS, decode_upload, resize_for_preview, save_image


# SSRF Protection Defaults
DEFAULT_MAX_BYTES = 80 * 1024 * 1024  # 80 MB
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 15.0
DEFAULT_MAX_REDIRECTS = 5

PROHIBITED_HOSTNAMES = {
    "localhost",
    "broadcasthost",
    "local",
    "internal",
    "lan",
    "home",
    "corp",
    "test",
    "invalid",
    "example",
}

GOOGLE_DRIVE_HOSTS = {
    "drive.google.com",
    "docs.google.com",
    "drive.usercontent.google.com",
}


class UrlIngestionError(Exception):
    """Base exception for all URL ingestion failures."""

    def __init__(self, message: str, error_code: str = "INGESTION_ERROR", recovery_hint: str = ""):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recovery_hint = recovery_hint


class InvalidUrlError(UrlIngestionError):
    def __init__(self, message: str = "Invalid URL syntax.", recovery_hint: str = "Check the URL format and try again."):
        super().__init__(message, "URL_INVALID", recovery_hint)


class UnsupportedSchemeError(UrlIngestionError):
    def __init__(self, scheme: str):
        super().__init__(
            f"Unsupported URL scheme '{scheme}'. Only HTTP and HTTPS are permitted.",
            "UNSUPPORTED_SCHEME",
            "Provide a standard http:// or https:// web address.",
        )


class SsrfBlockedError(UrlIngestionError):
    def __init__(self, detail: str = "Access to private, loopback, or internal network address is forbidden."):
        super().__init__(
            detail,
            "SSRF_BLOCKED",
            "Provide a publicly accessible Internet URL.",
        )


class DownloadTimeoutError(UrlIngestionError):
    def __init__(self, detail: str = "The remote server timed out."):
        super().__init__(
            detail,
            "DOWNLOAD_TIMEOUT",
            "The remote server was too slow to respond. Check the host or try again later.",
        )


class OversizedFileError(UrlIngestionError):
    def __init__(self, size_bytes: int, max_bytes: int = DEFAULT_MAX_BYTES):
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            f"The remote resource exceeds the maximum allowed size ({max_mb:.0f} MB).",
            "FILE_TOO_LARGE",
            f"Provide an image smaller than {max_mb:.0f} MB.",
        )


class InaccessibleHostError(UrlIngestionError):
    def __init__(self, message: str = "Unable to connect to the remote server."):
        super().__init__(
            message,
            "HOST_NOT_ACCESSIBLE",
            "Verify that the remote server is online and publicly reachable.",
        )


class GoogleDriveError(UrlIngestionError):
    def __init__(
        self,
        message: str = "Google Drive file is not publicly accessible.",
        recovery_hint: str = "Set the file sharing permissions to 'Anyone with the link can view/download' and try again.",
    ):
        super().__init__(message, "GOOGLE_DRIVE_INACCESSIBLE", recovery_hint)


class InvalidImageError(UrlIngestionError):
    def __init__(self, message: str = "The remote resource is not a supported image."):
        super().__init__(
            message,
            "NOT_AN_IMAGE",
            "Ensure the URL points directly to an image file (PNG, JPEG, TIFF, BMP, WEBP).",
        )


class UnsupportedFormatError(UrlIngestionError):
    def __init__(self, ext: str):
        super().__init__(
            f"Unsupported image format '{ext}'.",
            "UNSUPPORTED_FORMAT",
            f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


@dataclass
class UrlProvenance:
    source_type: str  # "URL" | "GOOGLE_DRIVE"
    original_url: str
    normalized_url: str
    filename: str
    content_type: str
    file_size_bytes: int
    sha256: str
    width: int
    height: int
    channels: int
    retrieval_timestamp: str
    drive_file_id: Optional[str] = None


@dataclass
class UrlIngestionResult:
    success: bool
    provenance: Optional[UrlProvenance] = None
    local_path: Optional[str] = None
    image_url: Optional[str] = None
    preview_url: Optional[str] = None
    preview_path: Optional[str] = None
    error_code: Optional[str] = None
    detail: Optional[str] = None
    recovery_hint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


def is_ip_prohibited(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str]:
    """Check if an IP address belongs to loopback, private, link-local, multicast, or reserved ranges."""
    # Check IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return is_ip_prohibited(ip.ipv4_mapped)

    if ip.is_loopback:
        return True, "Loopback address (127.0.0.0/8 or ::1) is forbidden."
    if ip.is_private:
        return True, "Private RFC 1918 / ULA network address is forbidden."
    if ip.is_link_local:
        return True, "Link-local address (169.254.0.0/16 or fe80::/10) is forbidden."
    if ip.is_multicast:
        return True, "Multicast address is forbidden."
    if ip.is_reserved:
        return True, "Reserved IP space is forbidden."
    if ip.is_unspecified:
        return True, "Unspecified address (0.0.0.0 or ::) is forbidden."

    # Explicit check for Carrier-grade NAT (100.64.0.0/10)
    try:
        cgnat = ipaddress.ip_network("100.64.0.0/10")
        if ip in cgnat:
            return True, "Shared address space (100.64.0.0/10) is forbidden."
    except Exception:
        pass

    # Explicit check for Cloud metadata service (169.254.169.254)
    if str(ip) == "169.254.169.254":
        return True, "Cloud metadata IP address is forbidden."

    return False, ""


def validate_url_syntax(url_str: str) -> tuple[str, str, int]:
    """Validate URL syntax, scheme, and extract hostname and port.

    Returns:
        (scheme, hostname, port)
    """
    cleaned = (url_str or "").strip()
    if not cleaned:
        raise InvalidUrlError("Empty URL provided.")

    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        raise InvalidUrlError(f"Malformed URL: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if not scheme:
        raise InvalidUrlError("URL is missing scheme (expected http:// or https://).")
    if scheme not in ("http", "https"):
        raise UnsupportedSchemeError(scheme)

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise InvalidUrlError("URL is missing a valid hostname.")

    # Disallow userinfo (e.g. http://user:pass@host)
    if parsed.username or parsed.password:
        raise InvalidUrlError("User credentials in URL authority are not allowed.")

    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, hostname, port


def validate_hostname_ssrf(hostname: str, port: int) -> None:
    """Resolve hostname and verify that no resolved IP addresses belong to prohibited ranges."""
    # Check for obvious internal or local domain names
    if hostname in PROHIBITED_HOSTNAMES or any(
        hostname.endswith(f".{d}") for d in PROHIBITED_HOSTNAMES
    ):
        raise SsrfBlockedError(f"Access to internal domain '{hostname}' is forbidden.")

    # Check if hostname is an IP literal
    try:
        ip_obj = ipaddress.ip_address(hostname)
        blocked, reason = is_ip_prohibited(ip_obj)
        if blocked:
            raise SsrfBlockedError(f"Blocked IP address '{hostname}': {reason}")
        return
    except ValueError:
        pass  # Not an IP literal, proceed to DNS resolution

    # Resolve hostname via DNS
    try:
        addrinfo = socket.getaddrinfo(hostname, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise InaccessibleHostError(f"Unable to resolve host '{hostname}': {exc}") from exc
    except Exception as exc:
        raise InaccessibleHostError(f"DNS lookup failure for '{hostname}': {exc}") from exc

    if not addrinfo:
        raise InaccessibleHostError(f"No IP addresses resolved for '{hostname}'.")

    # Verify every resolved IP
    for family, socktype, proto, canonname, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            blocked, reason = is_ip_prohibited(ip_obj)
            if blocked:
                raise SsrfBlockedError(f"Resolved address '{ip_str}' for host '{hostname}' is blocked: {reason}")
        except ValueError:
            raise SsrfBlockedError(f"Invalid resolved IP address '{ip_str}'.")


def extract_google_drive_file_id(url_str: str) -> Optional[str]:
    """Extract a Google Drive file ID from standard public sharing/download URL patterns."""
    parsed = urlparse(url_str.strip())
    host = (parsed.hostname or "").lower()
    if not (host in GOOGLE_DRIVE_HOSTS or host.endswith(".google.com")):
        return None

    path = parsed.path or ""
    # Pattern 1: /file/d/{ID}/...
    file_match = re.search(r"/file/d/([a-zA-Z0-9_-]{20,})", path)
    if file_match:
        return file_match.group(1)

    # Pattern 2: /open?id={ID} or /uc?id={ID} or query param 'id'
    query_params = parse_qs(parsed.query)
    if "id" in query_params:
        candidates = query_params["id"]
        for cand in candidates:
            if re.match(r"^[a-zA-Z0-9_-]{20,}$", cand):
                return cand

    # Pattern 3: /d/{ID}
    d_match = re.search(r"/d/([a-zA-Z0-9_-]{20,})", path)
    if d_match:
        return d_match.group(1)

    return None


def normalize_google_drive_url(file_id: str) -> str:
    """Convert an extracted Google Drive file ID to a normalized direct download URL."""
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"


def extract_google_drive_folder_id(url_str: str) -> Optional[str]:
    """Extract a Google Drive folder ID from public folder URL patterns."""
    try:
        parsed = urlparse(url_str.strip())
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if not (host in GOOGLE_DRIVE_HOSTS or host.endswith(".google.com")):
        return None

    path = parsed.path or ""
    # Pattern 1: /folders/{ID} or /drive/folders/{ID} or /drive/u/0/folders/{ID}
    folder_match = re.search(r"/folders/([a-zA-Z0-9_-]{20,})", path)
    if folder_match:
        return folder_match.group(1)

    # Pattern 2: /open?id={ID} (if it represents a folder)
    query_params = parse_qs(parsed.query)
    if "id" in query_params:
        candidates = query_params["id"]
        for cand in candidates:
            if re.match(r"^[a-zA-Z0-9_-]{20,}$", cand) and "folder" in path.lower():
                return cand

    return None


def enumerate_google_drive_folder(
    folder_id: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """Enumerate accessible image files within a public Google Drive folder.

    Returns:
        List of dicts: [{"file_id": id, "filename": name, "download_url": url, "source_type": "GOOGLE_DRIVE"}]
    """
    sess = session or requests.Session()
    sess.headers.update({
        "User-Agent": "SELENE-REG-X-VisionWorker/1.0 (Chandrayaan-2 Correspondence Scientific Workstation)",
    })

    validate_hostname_ssrf("drive.google.com", 443)

    folder_url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    try:
        resp = sess.get(folder_url, timeout=(connect_timeout, read_timeout), allow_redirects=False)
    except requests.exceptions.Timeout as exc:
        raise DownloadTimeoutError("Connection timed out while accessing Google Drive folder.") from exc
    except Exception as exc:
        raise InaccessibleHostError(f"Network error while accessing Google Drive folder: {exc}") from exc

    if resp.status_code in (301, 302, 303, 307, 308):
        loc = resp.headers.get("Location", "")
        if "accounts.google.com" in loc.lower():
            raise GoogleDriveError(
                "This Google Drive item is not publicly accessible.",
                "Set the folder sharing permissions to 'Anyone with the link can view' and try again.",
            )

    if resp.status_code in (401, 403):
        raise GoogleDriveError(
            "This Google Drive item is not publicly accessible.",
            "Set the folder sharing permissions to 'Anyone with the link can view' and try again.",
        )

    html = resp.text
    if any(x in html.lower() for x in ["sign in", "you need access", "access denied", "permission denied"]):
        raise GoogleDriveError(
            "This Google Drive item is not publicly accessible.",
            "Set the folder sharing permissions to 'Anyone with the link can view' and try again.",
        )


    discovered = []
    # Match items in embeddedfolderview
    entries = re.findall(r'id="entry-([a-zA-Z0-9_-]{20,})"[^>]*>.*?<div class="flip-entry-title">([^<]+)</div>', html, re.DOTALL)
    if not entries:
        entries = re.findall(r'/file/d/([a-zA-Z0-9_-]{20,})[^"]*".*?>([^<]+\.(?:png|jpg|jpeg|tif|tiff|bmp|webp))<', html, re.IGNORECASE)

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    seen_ids = set()
    for file_id, raw_name in entries:
        clean_fn = sanitize_filename(raw_name.strip())
        ext = Path(clean_fn).suffix.lower()
        if ext in image_exts and file_id not in seen_ids:
            seen_ids.add(file_id)
            discovered.append({
                "file_id": file_id,
                "filename": clean_fn,
                "download_url": normalize_google_drive_url(file_id),
                "source_type": "GOOGLE_DRIVE",
            })

    return discovered



def parse_content_disposition_filename(header_value: str) -> Optional[str]:
    """Extract filename safely from a Content-Disposition header."""
    if not header_value:
        return None
    # match filename*="utf-8''..." or filename="..."
    fn_star = re.search(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;]+)", header_value, re.IGNORECASE)
    if fn_star:
        raw = fn_star.group(1).strip("\"' ")
        return Path(raw).name
    fn = re.search(r'filename\s*=\s*"?([^";]+)"?', header_value, re.IGNORECASE)
    if fn:
        raw = fn.group(1).strip("\"' ")
        return Path(raw).name
    return None


def sanitize_filename(filename: str, fallback_ext: str = ".png") -> str:
    """Sanitize filename to prevent directory traversal and remove unsafe characters."""
    name = Path(filename).name
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._")
    if not clean:
        clean = f"downloaded_image_{int(time.time())}"
    ext = Path(clean).suffix.lower()
    if not ext or ext not in ALLOWED_EXTENSIONS:
        clean = f"{clean}{fallback_ext}"
    return clean


def infer_extension_from_bytes(data: bytes) -> str:
    """Detect image extension from magic byte signatures."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return ".tif"
    if data.startswith(b"BM"):
        return ".bmp"
    if data.startswith(b"RIFF") and b"WEBP" in data[8:16]:
        return ".webp"
    return ".png"


def check_google_interstitial(response_text: str, session: requests.Session, drive_file_id: str) -> Optional[str]:
    """Parse Google Drive's virus scan / confirmation interstitial page if present.

    Returns confirmed download URL if found, or None.
    """
    # Check for confirmation tokens
    # e.g., href="/uc?export=download&amp;confirm=XXXX..." or confirm=XXXX
    confirm_match = re.search(r'confirm=([0-9a-zA-Z_-]+)', response_text)
    if confirm_match:
        confirm_code = confirm_match.group(1)
        return f"https://drive.usercontent.google.com/download?id={drive_file_id}&export=download&confirm={confirm_code}"

    uuid_match = re.search(r'uuid=([0-9a-zA-Z_-]+)', response_text)
    if uuid_match:
        uuid = uuid_match.group(1)
        return f"https://drive.usercontent.google.com/download?id={drive_file_id}&export=download&confirm=t&uuid={uuid}"

    return None


def safe_download_image(
    url: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    session: Optional[requests.Session] = None,
) -> tuple[bytes, str, str, str]:
    """Download an image URL with strict SSRF protection, redirection validation, and streaming size limits.

    Returns:
        (image_bytes, final_url, content_type, suggested_filename)
    """
    sess = session or requests.Session()
    # Default User-Agent for standard scientific/HTTP fetching
    sess.headers.update({
        "User-Agent": "SELENE-REG-X-VisionWorker/1.0 (Chandrayaan-2 Correspondence Scientific Workstation)",
        "Accept": "image/*,*/*;q=0.8",
    })

    current_url = url
    redirect_count = 0

    while True:
        # Step 1: Validate URL syntax and SSRF on current hop
        scheme, hostname, port = validate_url_syntax(current_url)
        validate_hostname_ssrf(hostname, port)

        # Step 2: Request header / streaming body with allow_redirects=False
        try:
            resp = sess.get(
                current_url,
                stream=True,
                allow_redirects=False,
                timeout=(connect_timeout, read_timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise DownloadTimeoutError(f"Connection timed out while fetching {hostname}.") from exc
        except requests.exceptions.SSLError as exc:
            raise InaccessibleHostError(f"SSL/TLS verification failed for {hostname}: {exc}") from exc
        except requests.exceptions.ConnectionError as exc:
            raise InaccessibleHostError(f"Unable to connect to {hostname}: {exc}") from exc
        except Exception as exc:
            raise InaccessibleHostError(f"Network error while fetching URL: {exc}") from exc

        # Step 3: Check for redirects
        if resp.status_code in (301, 302, 303, 307, 308):
            redirect_count += 1
            if redirect_count > max_redirects:
                raise InaccessibleHostError(f"Too many redirects (exceeded limit of {max_redirects}).")

            location = resp.headers.get("Location")
            if not location:
                raise InaccessibleHostError("Redirect received without Location header.")

            next_url = urljoin(current_url, location)

            # Check if redirect is directing to Google Accounts login (private file)
            if "accounts.google.com" in next_url.lower():
                raise GoogleDriveError(
                    "Google Drive file is not publicly accessible. Set the file to 'Anyone with the link can view/download' and try again."
                )

            current_url = next_url
            continue

        # Step 4: Check HTTP status
        if resp.status_code == 401 or resp.status_code == 403:
            if "drive.google.com" in current_url or "drive.usercontent.google.com" in current_url:
                raise GoogleDriveError()
            raise InaccessibleHostError("The URL requires authentication or is not publicly accessible (HTTP 401/403).")

        if resp.status_code == 404:
            if "drive.google.com" in current_url or "drive.usercontent.google.com" in current_url:
                raise GoogleDriveError("Google Drive file not found or not accessible.", "Check the file link and sharing permissions.")
            raise InaccessibleHostError("Remote image not found (HTTP 404).")

        if resp.status_code >= 400:
            if "drive.google.com" in current_url or "drive.usercontent.google.com" in current_url:
                raise GoogleDriveError(
                    "Google Drive was unable to serve this file. Verify that the file exists and is set to 'Anyone with the link can view/download'.",
                    "Ensure sharing is set to 'Anyone with the link' and try again."
                )
            raise InaccessibleHostError(f"Remote server returned error HTTP {resp.status_code}.")

        # Step 5: Check Content-Length if provided
        cl = resp.headers.get("Content-Length")
        if cl:
            try:
                length = int(cl)
                if length > max_bytes:
                    raise OversizedFileError(length, max_bytes)
            except ValueError:
                pass

        # Step 6: Stream response body safely
        chunks: list[bytes] = []
        total_size = 0
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise OversizedFileError(total_size, max_bytes)
                chunks.append(chunk)
        except requests.exceptions.Timeout as exc:
            raise DownloadTimeoutError("Read timed out while downloading image body.") from exc
        except OversizedFileError:
            raise
        except Exception as exc:
            raise InaccessibleHostError(f"Failed to read image data: {exc}") from exc

        data = b"".join(chunks)
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        cd = resp.headers.get("Content-Disposition", "")
        suggested_filename = parse_content_disposition_filename(cd) or Path(urlparse(current_url).path).name

        return data, current_url, content_type, suggested_filename


def ingest_image_from_url(
    url_str: str,
    out_dir: Path | str,
    public_prefix: str = "/api/outputs/ingested",
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> UrlIngestionResult:
    """Main ingestion entrypoint.

    Validates URL, handles Google Drive normalization, enforces SSRF,
    downloads and verifies image bytes, creates safe disk artifacts, and returns provenance.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cleaned_url = (url_str or "").strip()
    if not cleaned_url:
        return UrlIngestionResult(
            success=False,
            error_code="URL_INVALID",
            detail="No URL was provided.",
            recovery_hint="Paste a valid HTTP/HTTPS image URL or Google Drive link.",
        )

    # 1. Google Drive Detection and Normalization
    session = requests.Session()
    drive_folder_id = extract_google_drive_folder_id(cleaned_url)
    drive_file_id = None

    try:
        if drive_folder_id:
            files = enumerate_google_drive_folder(drive_folder_id, session=session)
            if not files:
                raise GoogleDriveError(
                    "This Google Drive folder is empty or contains no supported image files.",
                    "Ensure the folder is public and contains image files.",
                )
            drive_file_id = files[0]["file_id"]
            source_type = "GOOGLE_DRIVE"
            fetch_url = files[0]["download_url"]
        else:
            drive_file_id = extract_google_drive_file_id(cleaned_url)
            source_type = "GOOGLE_DRIVE" if drive_file_id else "URL"
            fetch_url = normalize_google_drive_url(drive_file_id) if drive_file_id else cleaned_url

        data, final_url, content_type, suggested_fn = safe_download_image(
            fetch_url,
            max_bytes=max_bytes,
            session=session,
        )


        # 2. Handle Google Drive confirmation page if large file
        if drive_file_id and (len(data) < 50000 or content_type == "text/html"):
            prefix_head = data[:4000].decode("utf-8", errors="ignore").lower()
            if "<html" in prefix_head or "<!doctype html" in prefix_head:
                if any(x in prefix_head for x in ["sign in", "you need access", "access denied"]):
                    raise GoogleDriveError(
                        "Google Drive file is not publicly accessible. Set the file to 'Anyone with the link can view/download' and try again."
                    )
                confirm_url = check_google_interstitial(data.decode("utf-8", errors="ignore"), session, drive_file_id)
                if confirm_url:
                    data, final_url, content_type, suggested_fn = safe_download_image(
                        confirm_url,
                        max_bytes=max_bytes,
                        session=session,
                    )

        # 3. Validate content is not HTML / Text / JSON
        if len(data) == 0:
            raise InvalidImageError("The downloaded file is empty.")

        head_sniff = data[:512].decode("ascii", errors="ignore").lower().strip()
        if head_sniff.startswith("<!doctype html") or head_sniff.startswith("<html") or "<body" in head_sniff:
            if drive_file_id:
                raise GoogleDriveError(
                    "Google Drive file is not publicly accessible. Set the file to 'Anyone with the link can view/download' and try again."
                )
            raise InvalidImageError("The remote resource is an HTML webpage, not a raster image.")

        if head_sniff.startswith("{") and ("\"error\"" in head_sniff or "\"message\"" in head_sniff):
            raise InvalidImageError("The remote resource returned a JSON error response instead of an image.")

        # 4. Infer format and extension
        inferred_ext = infer_extension_from_bytes(data)
        safe_fn = sanitize_filename(suggested_fn or f"image{inferred_ext}", fallback_ext=inferred_ext)

        # 5. Scientific OpenCV Stack Validation
        try:
            image = decode_upload(data, safe_fn)
        except Exception as exc:
            raise InvalidImageError(f"Image validation failed: {exc}") from exc

        if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise InvalidImageError("The remote resource is corrupted or contains zero-dimension image data.")

        # 6. Save original image artifact
        image_file_path = out_path / safe_fn
        image_file_path.write_bytes(data)

        # 7. Generate web preview
        preview_fn = "preview.png"
        preview_file_path = out_path / preview_fn
        save_image(preview_file_path, resize_for_preview(image, max_side=1200))

        # 8. Compute checksum and provenance
        sha256 = hashlib.sha256(data).hexdigest()
        height, width = int(image.shape[0]), int(image.shape[1])
        channels = int(image.shape[2]) if image.ndim == 3 else 1
        now_iso = datetime.now(timezone.utc).isoformat()

        if not content_type or content_type == "application/octet-stream":
            content_type = mimetypes.guess_type(safe_fn)[0] or "image/png"

        provenance = UrlProvenance(
            source_type=source_type,
            original_url=cleaned_url,
            normalized_url=final_url,
            filename=safe_fn,
            content_type=content_type,
            file_size_bytes=len(data),
            sha256=sha256,
            width=width,
            height=height,
            channels=channels,
            retrieval_timestamp=now_iso,
            drive_file_id=drive_file_id,
        )

        public_prefix_clean = public_prefix.rstrip("/")
        image_url = f"{public_prefix_clean}/{safe_fn}"
        preview_url = f"{public_prefix_clean}/{preview_fn}"

        return UrlIngestionResult(
            success=True,
            provenance=provenance,
            local_path=str(image_file_path.resolve()),
            image_url=image_url,
            preview_url=preview_url,
            preview_path=preview_url,
        )

    except UrlIngestionError as err:
        return UrlIngestionResult(
            success=False,
            error_code=err.error_code,
            detail=err.message,
            recovery_hint=err.recovery_hint,
        )
    except Exception as exc:
        return UrlIngestionResult(
            success=False,
            error_code="SERVER_ERROR",
            detail=f"Ingestion failed unexpectedly: {exc}",
            recovery_hint="Check server logs or try another image source.",
        )


def cleanup_old_ingestions(base_dir: Path | str, max_age_seconds: int = 7200, max_items: int = 50) -> int:
    """Clean up old temporary ingestion directories to prevent unbounded accumulation.

    Only operates inside designated 'ingested' directory.
    """
    root = Path(base_dir)
    if not root.exists() or not root.is_dir() or root.name != "ingested":
        return 0

    now = time.time()
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime)

    removed = 0
    for p in dirs:
        try:
            age = now - p.stat().st_mtime
            if age > max_age_seconds or len(dirs) - removed > max_items:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except Exception:
            pass

    return removed
