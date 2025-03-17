import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def per_tensor_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_amax = x.abs().float().amax().clamp(1e-4)
    return (x * (448.0 / x_amax)).to(torch.float8_e4m3fn), x_amax / 448.0

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()
    
    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    # LHS row-major, RHS column-major
    # LHS SF col-major (TMA 16B aligned), RHS SF row-major
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def compare_construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()
    
    # per-tensor fp8 mma
    x_raw_fp8, x_sf = per_tensor_cast_to_fp8(x)
    y_raw_fp8, y_sf = per_tensor_cast_to_fp8(y)
    out_raw = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    # print(f'{x_raw_fp8.shape=}, {y_raw_fp8.shape=}, {x_sf.shape=}, {y_sf.shape=}, {out_raw.shape=}')
    torch._scaled_mm(x_raw_fp8, y_raw_fp8.t(), scale_a=x_sf, scale_b=y_sf, out_dtype=torch.bfloat16, out=out_raw)
    raw_diff = calc_diff(out_raw, ref_out)

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    # LHS row-major, RHS column-major
    # LHS SF col-major (TMA 16B aligned), RHS SF row-major
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
    dg_diff = calc_diff(out, ref_out)
    
    x_triton, y_triton = deep_gemm.per_token_quantize(x), deep_gemm.per_block_quantize(y)
    out_triton = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_triton, y_triton, out_triton)
    triton_diff = calc_diff(out_triton, ref_out)
    
    print(f' > {m=}, {k=}, {n=}, per-tensor FP8 MMA: {raw_diff:.5f}, DeepGemm: {dg_diff:.5f}, Triton: {triton_diff:.5f}')

def benchmark(m: int, k: int, n: int) -> None:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    out_ref = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    # per-tensor fp8 mma
    x_raw_fp8, x_sf = per_tensor_cast_to_fp8(x)
    y_raw_fp8, y_sf = per_tensor_cast_to_fp8(y)
    out_raw = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    
    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    # LHS row-major, RHS column-major
    # LHS SF col-major (TMA 16B aligned), RHS SF row-major
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    
    def test_bf16():
        torch.matmul(x, y.t(), out=out_ref)
    
    def test_per_tensor_fp8():
        torch._scaled_mm(x_raw_fp8, y_raw_fp8.t(), scale_a=x_sf, scale_b=y_sf, out_dtype=torch.bfloat16, out=out_raw)
    
    def test_dg_fp8():
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
    
    def benchmark_func(test_func, repeat=8):
        # warmup        
        for _ in range(2):
            test_func()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            test_func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat
    
    bf16_time = benchmark_func(test_bf16)
    bf16_perf = 2 * m * n * k / (bf16_time * 1e-3) / 1e12
    per_tensor_fp8_time = benchmark_func(test_per_tensor_fp8)
    fp8_perf = 2 * m * n * k / (per_tensor_fp8_time * 1e-3) / 1e12
    dg_fp8_time = benchmark_func(test_dg_fp8)
    dg_fp8_perf = 2 * m * n * k / (dg_fp8_time * 1e-3) / 1e12
    
    print(f' > {m=}, {k=}, {n=}, BF16: {bf16_perf:.2f} TFLOPS, Per-tensor FP8: {fp8_perf:.2f} TFLOPS, DeepGemm FP8: {dg_fp8_perf:.2f} TFLOPS')        

def test_gemm() -> None:
    print('Testing GEMM:')
    for scale in [1, 2, 4, 8, 16]:
        m, k, n = 1024 * scale, 4096 * scale, 1024 * scale
        benchmark(m, k, n)
        for _ in range(5):
            compare_construct(m, k, n)
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
