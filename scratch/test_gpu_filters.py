import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time

def test_gpu_gaussian():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    device = torch.device("cuda:0")
    
    # 1D Gaussian kernel
    sigma = 15.0
    radius = int(round(3.0 * sigma))
    ksize = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)

    img = np.random.rand(500, 500).astype(np.float32)
    t_in = torch.as_tensor(img, device=device).unsqueeze(0).unsqueeze(0)
    
    t0 = time.perf_counter()
    # Separable conv
    pad_h = radius
    pad_w = radius
    # Pad reflect
    padded = F.pad(t_in, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    blurred_x = F.conv2d(padded, kx)
    blurred_xy = F.conv2d(blurred_x, ky)
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t0) * 1000.0
    
    out_gpu = blurred_xy.squeeze().cpu().numpy()
    
    t0 = time.perf_counter()
    out_cpu = cv2.GaussianBlur(img, (0, 0), sigma)
    t_cpu = (time.perf_counter() - t0) * 1000.0
    
    diff = np.max(np.abs(out_gpu - out_cpu))
    print(f"Gaussian Blur: max diff={diff:.6f}, GPU={t_gpu:.2f}ms, CPU={t_cpu:.2f}ms")
    assert diff < 1e-2

if __name__ == "__main__":
    test_gpu_gaussian()
