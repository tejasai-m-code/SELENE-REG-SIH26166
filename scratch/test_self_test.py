import sys
from pathlib import Path
sys.path.insert(0, str(Path("artifacts/api-server/python").resolve()))

import json
from time import perf_counter

def run_test():
    import torch
    cuda_detected = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    if not cuda_detected:
        return {"cuda_device_detected": "FAIL", "overall_status": "FAIL"}
    
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    
    t0 = perf_counter()
    a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    b = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    alloc_pass = (a.is_cuda and b.is_cuda)
    
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    sync_pass = True
    compute_pass = bool(c.shape == (1024, 1024) and torch.isfinite(c).all().item())
    dt_ms = (perf_counter() - t0) * 1000.0
    
    alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    res_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
    tot_mb = props.total_memory / (1024 * 1024)
    
    del a, b, c
    torch.cuda.empty_cache()
    
    return {
        "cuda_device_detected": "PASS" if cuda_detected else "FAIL",
        "tensor_allocation": "PASS" if alloc_pass else "FAIL",
        "gpu_computation": "PASS" if compute_pass else "FAIL",
        "synchronization": "PASS" if sync_pass else "FAIL",
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_total_mb": round(tot_mb, 1),
        "vram_allocated_mb": round(alloc_mb, 2),
        "vram_reserved_mb": round(res_mb, 2),
        "vram_free_mb": round(tot_mb - res_mb, 1),
        "compute_time_ms": round(dt_ms, 2),
        "overall_status": "PASS" if (cuda_detected and alloc_pass and compute_pass and sync_pass) else "FAIL",
        "execution_mode": "GPU / HYBRID",
    }

print(json.dumps(run_test(), indent=2))
