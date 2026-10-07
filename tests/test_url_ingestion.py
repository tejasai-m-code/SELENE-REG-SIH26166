"""Tests for URL image ingestion layer.

Verifies all 15 required ingestion and SSRF protection requirements:
1. Valid HTTPS URL ingestion
2. Invalid URL syntax rejection
3. Unsupported scheme rejection
4. Download timeout handling
5. Non-image content rejection
6. Oversized response rejection
7. Inaccessible host handling
8. Localhost rejection
9. Private IP rejection
10. Redirect to blocked destination revalidation
11. Public Google Drive URL extraction and normalization
12. Inaccessible / private Google Drive response handling
13. Local upload regression verification
14. API response contract compliance
15. Provenance metadata completeness
"""

import io
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import requests

from app.services.url_ingestion import (
    DEFAULT_MAX_BYTES,
    DownloadTimeoutError,
    GoogleDriveError,
    InaccessibleHostError,
    InvalidImageError,
    InvalidUrlError,
    OversizedFileError,
    SsrfBlockedError,
    UnsupportedFormatError,
    UnsupportedSchemeError,
    UrlIngestionResult,
    UrlProvenance,
    extract_google_drive_file_id,
    ingest_image_from_url,
    is_ip_prohibited,
    normalize_google_drive_url,
    safe_download_image,
    validate_hostname_ssrf,
    validate_url_syntax,
)
from app.utils.image_utils import decode_upload


def _create_test_png(width: int = 120, height: int = 120, color: int = 128) -> bytes:
    """Helper to generate valid PNG bytes in memory."""
    img = np.full((height, width), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestUrlIngestionSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="selene_test_ingest_"))
        self.sample_png = _create_test_png(100, 100, color=180)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Valid HTTPS URL
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_01_valid_https_url(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "Content-Type": "image/png",
            "Content-Length": str(len(self.sample_png)),
            "Content-Disposition": 'inline; filename="lunar_crater.png"',
        }
        mock_resp.iter_content.return_value = [self.sample_png]
        mock_get.return_value = mock_resp

        result = ingest_image_from_url(
            "https://example.com/images/lunar_crater.png",
            out_dir=self.temp_dir,
            public_prefix="/api/outputs/ingested/test1",
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.provenance)
        self.assertEqual(result.provenance.source_type, "URL")
        self.assertEqual(result.provenance.filename, "lunar_crater.png")
        self.assertEqual(result.provenance.width, 100)
        self.assertEqual(result.provenance.height, 100)
        self.assertEqual(result.provenance.file_size_bytes, len(self.sample_png))
        self.assertTrue(Path(result.local_path).exists())

    # 2. Invalid URL
    def test_02_invalid_url_syntax(self):
        invalid_urls = ["", "   ", "not_a_url", "http://", "https:///empty-host"]
        for bad_url in invalid_urls:
            result = ingest_image_from_url(bad_url, out_dir=self.temp_dir)
            self.assertFalse(result.success)
            self.assertEqual(result.error_code, "URL_INVALID")

    # 3. Unsupported scheme
    def test_03_unsupported_scheme(self):
        unsupported = [
            "ftp://example.com/test.png",
            "file:///etc/passwd",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
            "javascript:alert(1)",
            "gopher://example.com/",
        ]
        for url in unsupported:
            result = ingest_image_from_url(url, out_dir=self.temp_dir)
            self.assertFalse(result.success)
            self.assertEqual(result.error_code, "UNSUPPORTED_SCHEME")

    # 4. Download timeout
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_04_download_timeout(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = ingest_image_from_url("https://example.com/slow_image.png", out_dir=self.temp_dir)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "DOWNLOAD_TIMEOUT")

    # 5. Non-image content
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_05_non_image_content_rejected(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        html_content = b"<!DOCTYPE html><html><body><h1>404 Not an image</h1></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.iter_content.return_value = [html_content]
        mock_get.return_value = mock_resp

        result = ingest_image_from_url("https://example.com/page.html", out_dir=self.temp_dir)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "NOT_AN_IMAGE")

    # 6. Oversized response
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_06_oversized_response_rejected(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        # Simulate small limit for testing
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/png", "Content-Length": "200000000"}
        mock_resp.iter_content.return_value = [b"A" * 1024 * 1024] * 10
        mock_get.return_value = mock_resp

        result = ingest_image_from_url(
            "https://example.com/giant.png",
            out_dir=self.temp_dir,
            max_bytes=5 * 1024 * 1024,  # 5 MB test limit
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "FILE_TOO_LARGE")

    # 7. Inaccessible host
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_07_inaccessible_host(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = ingest_image_from_url("https://example.com/offline.png", out_dir=self.temp_dir)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "HOST_NOT_ACCESSIBLE")

    # 8. Localhost rejection
    def test_08_localhost_rejection(self):
        localhost_urls = [
            "http://localhost/image.png",
            "http://127.0.0.1/image.png",
            "http://127.0.0.254/test.png",
            "http://[::1]/image.png",
            "https://localhost:8443/image.tif",
        ]
        for url in localhost_urls:
            result = ingest_image_from_url(url, out_dir=self.temp_dir)
            self.assertFalse(result.success)
            self.assertEqual(result.error_code, "SSRF_BLOCKED")

    # 9. Private IP rejection
    @patch("socket.getaddrinfo")
    def test_09_private_ip_rejection(self, mock_getaddrinfo):
        private_ips = [
            ("10.0.0.1", "RFC 1918 10/8"),
            ("172.16.5.10", "RFC 1918 172.16/12"),
            ("192.168.1.1", "RFC 1918 192.168/16"),
            ("169.254.169.254", "Link-local / AWS metadata"),
            ("100.64.0.1", "Carrier-grade NAT"),
            ("fc00::1", "IPv6 Unique Local"),
            ("fe80::1", "IPv6 Link-Local"),
            ("224.0.0.1", "Multicast"),
        ]
        for ip, label in private_ips:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))
            ]
            result = ingest_image_from_url(f"http://internal-host-{label.replace(' ', '_')}/img.png", out_dir=self.temp_dir)
            self.assertFalse(result.success, f"Failed to block {ip} ({label})")
            self.assertEqual(result.error_code, "SSRF_BLOCKED")

    # 10. Redirect to blocked destination
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_10_redirect_to_blocked_destination(self, mock_get, mock_getaddrinfo):
        # First request resolves to safe public IP
        # But server responds with 302 Location: http://169.254.169.254/latest/meta-data
        mock_getaddrinfo.side_effect = [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "http://169.254.169.254/latest/meta-data"}
        mock_get.return_value = mock_resp

        result = ingest_image_from_url("http://example.com/redirect-to-metadata", out_dir=self.temp_dir)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "SSRF_BLOCKED")

    # 11. Public Google Drive URL normalization
    def test_11_google_drive_url_normalization(self):
        urls = [
            ("https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view?usp=sharing", "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"),
            ("https://drive.google.com/open?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms", "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"),
            ("https://drive.google.com/uc?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms&export=download", "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"),
            ("https://docs.google.com/uc?export=download&id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms", "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"),
        ]
        for url, expected_id in urls:
            extracted = extract_google_drive_file_id(url)
            self.assertEqual(extracted, expected_id)
            normalized = normalize_google_drive_url(extracted)
            self.assertIn(expected_id, normalized)
            self.assertTrue(normalized.startswith("https://drive.usercontent.google.com/download"))

    # 12. Inaccessible / private Google Drive response
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_12_private_google_drive_response(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.190.46", 443))
        ]
        # Simulate Google returning redirect to Google Accounts sign-in
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {
            "Location": "https://accounts.google.com/v3/signin/identifier?dsh=S123"
        }
        mock_get.return_value = mock_resp

        drive_url = "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view"
        result = ingest_image_from_url(drive_url, out_dir=self.temp_dir)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "GOOGLE_DRIVE_INACCESSIBLE")
        self.assertIn("Anyone with the link", result.recovery_hint)

    # 13. Local upload regression
    def test_13_local_upload_regression(self):
        # Ensure decode_upload continues to decode supported formats properly
        decoded = decode_upload(self.sample_png, "test.png")
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (100, 100))
        self.assertEqual(decoded.dtype, np.uint8)

    # 14. API response contract
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_14_api_response_contract(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.iter_content.return_value = [self.sample_png]
        mock_get.return_value = mock_resp

        result = ingest_image_from_url("https://example.com/test.png", out_dir=self.temp_dir)
        d = result.to_dict()

        self.assertIn("success", d)
        self.assertTrue(d["success"])
        self.assertIn("provenance", d)
        self.assertIn("local_path", d)
        self.assertIn("preview_path", d)

    # 15. Provenance metadata completeness
    @patch("socket.getaddrinfo")
    @patch.object(requests.Session, "get")
    def test_15_provenance_metadata(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.190.46", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "Content-Type": "image/png",
            "Content-Disposition": 'attachment; filename="ohrc_crater.png"',
        }
        mock_resp.iter_content.return_value = [self.sample_png]
        mock_get.return_value = mock_resp

        drive_url = "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view"
        result = ingest_image_from_url(drive_url, out_dir=self.temp_dir)

        self.assertTrue(result.success)
        prov = result.provenance
        self.assertEqual(prov.source_type, "GOOGLE_DRIVE")
        self.assertEqual(prov.drive_file_id, "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        self.assertEqual(prov.filename, "ohrc_crater.png")
        self.assertEqual(prov.content_type, "image/png")
        self.assertEqual(prov.file_size_bytes, len(self.sample_png))
        self.assertIsNotNone(prov.sha256)
        self.assertEqual(len(prov.sha256), 64)
        self.assertIsNotNone(prov.retrieval_timestamp)
        self.assertEqual(prov.width, 100)
        self.assertEqual(prov.height, 100)


class TestGoogleDriveFolderIngestion(unittest.TestCase):
    """Google Drive folder URL parsing, private detection, and file enumeration."""

    def test_folder_id_extraction(self):
        from app.services.url_ingestion import extract_google_drive_folder_id
        url = "https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0jklmnopqrstuvwxyz"
        folder_id = extract_google_drive_folder_id(url)
        self.assertEqual(folder_id, "1a2b3c4d5e6f7g8h9i0jklmnopqrstuvwxyz")

    @patch("requests.Session.get")
    def test_private_drive_folder_rejected(self, mock_get):
        from app.services.url_ingestion import ingest_image_from_url
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "https://accounts.google.com/signin/v2/identifier"}
        mock_get.return_value = mock_resp

        folder_url = "https://drive.google.com/drive/folders/1PrivateFolderIdLongEnough123456"
        res = ingest_image_from_url(folder_url, out_dir=tempfile.mkdtemp())
        self.assertFalse(res.success)
        self.assertIn("not publicly accessible", res.detail.lower())


if __name__ == "__main__":
    unittest.main()

