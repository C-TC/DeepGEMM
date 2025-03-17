from typing import Tuple
import torch
from deep_gemm.jit_kernels.utils import get_col_major_tma_aligned_tensor
import triton
from triton import language as tl

@triton.jit
def _per_token_quantize(
    x_ptr,
    x_fp8_ptr,
    x_sf_ptr,
    m, # row
    n, # column
    m_stride, 
    n_stride,
    eps,
    fp8_max,
    BLOCK_SIZE: tl.constexpr # number of 128-element blocks
):
    # input x (m,n) can be row/column-major
    # output x_fp8 (m,n) is row-major
    # output x_sf (m,n//128) is column-major, alignment requirement 16 bytes in m-axis
    
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    off_r = pid_r * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_c = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = off_r[:, None] * m_stride + off_c[None, :] * n_stride
    mask = (off_r[:, None] < m) & (off_c[None, :] < n)
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(x).to(tl.float32), axis=-1), eps)
    x_sf = (amax / fp8_max).reshape(BLOCK_SIZE)
    x_fp8 = (x * (fp8_max / amax).expand_dims(axis=1)).to(x_fp8_ptr.dtype.element_ty)
    offset_output = off_r[:, None] * n + off_c[None, :]
    offset_sf = off_r + m * pid_c
    tl.store(x_fp8_ptr + offset_output, x_fp8, mask=mask)
    tl.store(x_sf_ptr + offset_sf, x_sf)
    
def per_token_quantize(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-4,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:    
    # input x (m,n) can be row/column-major
    # output x_fp8 (m,n) is row-major
    # output x_sf (m,n//128) is column-major
    assert block_size == 128, 'Currently only support block_size=128'
    assert x.dim() == 2 and x.size(1) % block_size == 0
    m, n = x.shape
    m_stride, n_stride = x.stride(0), x.stride(1)
    
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    
    x_fp8 = torch.empty((m, n), dtype=dtype, device=x.device)
    x_sf = torch.empty((n // block_size, m), dtype=torch.float32, device=x.device).transpose(0, 1)
    _per_token_quantize[(m // block_size, n // block_size)](
        x,
        x_fp8,
        x_sf,
        m,
        n,
        m_stride,
        n_stride,
        eps,
        fp8_max,
        block_size
    )
    assert x_fp8.stride(0) == n and x_fp8.stride(1) == 1 # row-major
    assert x_sf.stride(0) == 1 and x_sf.stride(1) == m # column-major
    
    return x_fp8, x_sf

def transpose_per_token_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    return per_token_quantize(x.transpose(0, 1))

@triton.jit
def _per_block_quantize(
    x_ptr,
    x_fp8_ptr,
    x_sf_ptr,
    m, # row
    n, # column
    m_stride,
    n_stride,
    eps,
    fp8_max,
    BLOCK_SIZE: tl.constexpr
):
    # input x (m,n) can be row/column-major
    # output x_fp8 (m,n) is row-major
    # output x_sf (m//128,n//128) is row-major
    
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    off_r = pid_r * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_c = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = off_r[:, None] * m_stride + off_c[None, :] * n_stride
    mask = (off_r[:, None] < m) & (off_c[None, :] < n)
    x_tile = tl.load(x_ptr + offset, mask=mask, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(x_tile).to(tl.float32)), eps)
    x_fp8 = (x_tile * (fp8_max / amax)).to(x_fp8_ptr.dtype.element_ty)
    offset_fp8 = off_r[:, None] * n + off_c[None, :]
    offset_sf = pid_r * n // BLOCK_SIZE + pid_c
    tl.store(x_fp8_ptr + offset_fp8, x_fp8, mask=mask)
    tl.store(x_sf_ptr + offset_sf, amax / fp8_max)

def per_block_quantize(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-4,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input x (m,n) can be row/column-major
    # output x_fp8 (m,n) is row-major
    # output x_sf (m//128,n//128) is row-major
    assert block_size == 128, 'Currently only support block_size=128'
    assert x.dim() == 2 and x.size(0) % block_size == 0 and x.size(1) % block_size == 0
    m, n = x.shape
    m_stride, n_stride = x.stride(0), x.stride(1)
    
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    
    x_fp8 = torch.empty((m, n), dtype=dtype, device=x.device)
    x_sf = torch.empty((m // block_size, n // block_size), dtype=torch.float32, device=x.device)
    _per_block_quantize[(m // block_size, n // block_size)](
        x,
        x_fp8,
        x_sf,
        m,
        n,
        m_stride,
        n_stride,
        eps,
        fp8_max,
        block_size
    )
    assert x_fp8.stride(0) == n and x_fp8.stride(1) == 1 # row-major
    assert x_sf.stride(0) == n // block_size and x_sf.stride(1) == 1 # row-major
    return x_fp8, x_sf
    
def transpose_per_block_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    return per_block_quantize(x.transpose(0, 1))


def print_to_csv_file(x: torch.Tensor, name: str):
    x = x.to(torch.float32)
    with open(name, 'w') as f:
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                f.write(f'{x[i, j]:.4f},')
            f.write('\n')
    
@torch.compile
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (fp8_max / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), get_col_major_tma_aligned_tensor((x_amax / fp8_max).view(m, -1))

@torch.compile
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    x_padded = torch.zeros((m, n), dtype=x.dtype, device=x.device)
    x_padded = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (fp8_max / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded).contiguous(), (x_amax / fp8_max).view(x_view.size(0), x_view.size(2))

def test_per_token_quantize():
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    x_fp8, x_sf = per_token_quantize(x)
    assert x_fp8.shape == (1024, 1024), f'{x_fp8.shape=}'
    assert x_sf.shape == (1024, 8), f'{x_sf.shape=}'
    ref_x_fp8, ref_x_sf = per_token_cast_to_fp8(x)
    x_max_err = torch.max(torch.abs(x_fp8.to(torch.float32) - ref_x_fp8.to(torch.float32)))
    sf_max_err = torch.max(torch.abs(x_sf.to(torch.float32) - ref_x_sf.to(torch.float32)))
    print(f'per_token_quantize x_max_err={x_max_err:.4f}, sf_max_err={sf_max_err:.4f}')

def test_transpose_per_token_quantize():
    
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    x_fp8, x_sf = transpose_per_token_quantize(x)
    assert x_fp8.shape == (1024, 1024), f'{x_fp8.shape=}'
    assert x_sf.shape == (1024, 8), f'{x_sf.shape=}'
    ref_x_fp8, ref_x_sf = per_token_cast_to_fp8(x.transpose(0, 1).contiguous())
    x_max_err = torch.max(torch.abs(x_fp8.to(torch.float32) - ref_x_fp8.to(torch.float32)))
    sf_max_err = torch.max(torch.abs(x_sf.to(torch.float32) - ref_x_sf.to(torch.float32)))
    print(f'transpose_per_token_quantize x_max_err={x_max_err:.4f}, sf_max_err={sf_max_err:.4f}')

def test_per_block_quantize():
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    x_fp8, x_sf = per_block_quantize(x)
    assert x_fp8.shape == (1024, 1024), f'{x_fp8.shape=}'
    assert x_sf.shape == (8, 8), f'{x_sf.shape=}'
    ref_x_fp8, ref_x_sf = per_block_cast_to_fp8(x)
    x_max_err = torch.max(torch.abs(x_fp8.to(torch.float32) - ref_x_fp8.to(torch.float32)))
    sf_max_err = torch.max(torch.abs(x_sf.to(torch.float32) - ref_x_sf.to(torch.float32)))
    print(f'per_block_quantize x_max_err={x_max_err:.4f}, sf_max_err={sf_max_err:.4f}')
    
def test_transpose_per_block_quantize():
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    x_fp8, x_sf = transpose_per_block_quantize(x)
    assert x_fp8.shape == (1024, 1024), f'{x_fp8.shape=}'
    assert x_sf.shape == (8, 8), f'{x_sf.shape=}'
    ref_x_fp8, ref_x_sf = per_block_cast_to_fp8(x.transpose(0, 1).contiguous())
    x_max_err = torch.max(torch.abs(x_fp8.to(torch.float32) - ref_x_fp8.to(torch.float32)))
    sf_max_err = torch.max(torch.abs(x_sf.to(torch.float32) - ref_x_sf.to(torch.float32)))
    print(f'transpose_per_block_quantize x_max_err={x_max_err:.4f}, sf_max_err={sf_max_err:.4f}')

def benchmark():
    size = [2 ** i for i in range(10, 15)]
    for s in size:
        x = torch.randn(s, s, device='cuda', dtype=torch.bfloat16)
        
        for _ in range(2):
            # warmup
            per_token_quantize(x)
            per_block_quantize(x)
            per_token_cast_to_fp8(x)
            per_block_cast_to_fp8(x)
        
        def bench(func, x):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0.record()
            for _ in range(32):
                func(x)
            t1.record()
            torch.cuda.synchronize()
            return t0.elapsed_time(t1) / 32

        print(f'{s=}, triton_per_token: {bench(lambda x: per_token_quantize(x), x):.4f}, torch_per_token: {bench(lambda x: per_token_cast_to_fp8(x), x):.4f}, triton_per_block: {bench(lambda x: per_block_quantize(x), x):.4f}, torch_per_block: {bench(lambda x: per_block_cast_to_fp8(x), x):.4f}')
        

def benchmark_transpose():
    size = [2 ** i for i in range(10, 15)]
    for s in size:
        x = torch.randn(s, s, device='cuda', dtype=torch.bfloat16)
        
        for _ in range(2):
            # warmup
            transpose_per_token_quantize(x)
            transpose_per_block_quantize(x)
            per_token_cast_to_fp8(x.transpose(0, 1).contiguous())
            per_block_cast_to_fp8(x.transpose(0, 1).contiguous())
        
        def bench(func, x):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0.record()
            for _ in range(32):
                func(x)
            t1.record()
            torch.cuda.synchronize()
            return t0.elapsed_time(t1) / 32

        print(f'{s=}, triton_per_token: {bench(lambda x: transpose_per_token_quantize(x), x):.4f}, torch_per_token: {bench(lambda x: per_token_cast_to_fp8(x.transpose(0, 1).contiguous()), x):.4f}, triton_per_block: {bench(lambda x: transpose_per_block_quantize(x), x):.4f}, torch_per_block: {bench(lambda x: per_block_cast_to_fp8(x.transpose(0, 1).contiguous()), x):.4f}')
        
        


if __name__ == "__main__":

        
    test_per_token_quantize()
    test_per_block_quantize()
    test_transpose_per_token_quantize()
    test_transpose_per_block_quantize()
    
    benchmark()
    benchmark_transpose()
    