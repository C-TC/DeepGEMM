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
    num_rows,
    eps,
    fp8_max,
    BLOCK_SIZE: tl.constexpr # number of 128-element blocks
):
    pid = tl.program_id(0)
     
    off_r = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    off_c = tl.arange(0, BLOCK_SIZE)
    offset = off_r[:, None] * BLOCK_SIZE + off_c[None, :]
    mask = (off_r[:, None] < num_rows) & (off_c[None, :] < BLOCK_SIZE)
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    
    amax = tl.maximum(tl.max(tl.abs(x).to(tl.float32), axis=-1), eps)
    x_sf = (amax / fp8_max).reshape(BLOCK_SIZE)
    x_fp8 = (x * (fp8_max / amax).expand_dims(axis=1)).to(x_fp8_ptr.dtype.element_ty)

    tl.store(x_fp8_ptr + offset, x_fp8, mask=mask)
    
    tl.store(x_sf_ptr + off_r, x_sf, mask=(off_r < num_rows))
    
def per_token_quantize(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-4,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:    
    assert block_size == 128, 'Currently only support block_size=128'
    assert x.dim() in [2, 3] and x.size(-1) % block_size == 0
    assert x.is_contiguous()
    
    if x.dim() == 3:
        mb_size = x.size(0)
        seq_len = x.size(1)
    else:
        mb_size = None
        seq_len = x.size(0)
    
    x = x.view(-1, block_size)
    
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    
    x_fp8 = torch.empty(x.shape, dtype=dtype, device=x.device)
    x_sf = torch.empty((x.numel()//block_size), dtype=torch.float32, device=x.device)
    _per_token_quantize[(x.numel()//(block_size ** 2),)](
        x,
        x_fp8,
        x_sf,
        x.numel()//block_size,
        eps,
        fp8_max,
        block_size
    )
    if mb_size is None:
        return x_fp8.view(seq_len, -1), get_col_major_tma_aligned_tensor(x_sf.view(seq_len, -1))
    else:
        return x_fp8.view(mb_size, seq_len, -1), get_col_major_tma_aligned_tensor(x_sf.view(mb_size, seq_len, -1))

def transpose_per_token_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    x = x.transpose(-1, -2).contiguous()
    return per_token_quantize(x)


@triton.jit
def _per_block_quantize(
    x_ptr,
    x_fp8_ptr,
    x_sf_ptr,
    num_rows,
    num_cols,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_SIZE: tl.constexpr # number of 128x128-element blocks
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    off_r = pid_r * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_c = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = off_r[:, None] * num_cols + off_c[None, :]
    mask = (off_r[:, None] < num_rows) & (off_c[None, :] < num_cols)
    x_tile = tl.load(x_ptr + offset, mask=mask, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(x_tile).to(tl.float32)), eps)
    x_fp8 = (x_tile * (fp8_max / amax)).to(x_fp8_ptr.dtype.element_ty)
    tl.store(x_fp8_ptr + offset, x_fp8, mask=mask)
    
    n_blocks_per_row = tl.cdiv(num_cols, BLOCK_SIZE)
    tl.store(x_sf_ptr + pid_r * n_blocks_per_row + pid_c, amax / fp8_max)

def per_block_quantize(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-4,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert block_size == 128, 'Currently only support block_size=128'
    assert x.dim() in [2, 3] and x.size(-1) % block_size == 0
    assert x.is_contiguous()
    
    if x.dim() == 3:
        mb_size = x.size(0)
        seq_len = x.size(1)
    else:
        mb_size = None
        seq_len = x.size(0)
    
    x = x.view(-1, x.size(-1))
    
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    
    x_fp8 = torch.empty(x.shape, dtype=dtype, device=x.device)
    x_sf = torch.empty((x.numel() // (block_size ** 2)), dtype=torch.float32, device=x.device)
    _per_block_quantize[(x.size(0) // block_size, x.size(1) // block_size)](
        x,
        x_fp8,
        x_sf,
        x.size(0),
        x.size(1),
        eps,
        fp8_min,
        fp8_max,
        block_size
    )
    if mb_size is None:
        return x_fp8.view(seq_len, -1), x_sf.view(seq_len // block_size, -1)
    else:
        return x_fp8.view(mb_size, seq_len, -1), x_sf.view(mb_size, seq_len // block_size, -1)
    
def transpose_per_block_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    x = x.transpose(-1, -2).contiguous()
    return per_block_quantize(x)


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

def test_per_block_quantize():
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    x_fp8, x_sf = per_block_quantize(x)
    assert x_fp8.shape == (1024, 1024), f'{x_fp8.shape=}'
    assert x_sf.shape == (8, 8), f'{x_sf.shape=}'
    ref_x_fp8, ref_x_sf = per_block_cast_to_fp8(x)
    # print_to_csv_file(x_fp8, 'x_fp8.csv')
    # print_to_csv_file(ref_x_fp8, 'ref_x_fp8.csv')
    x_max_err = torch.max(torch.abs(x_fp8.to(torch.float32) - ref_x_fp8.to(torch.float32)))
    sf_max_err = torch.max(torch.abs(x_sf.to(torch.float32) - ref_x_sf.to(torch.float32)))
    print(f'per_block_quantize x_max_err={x_max_err:.4f}, sf_max_err={sf_max_err:.4f}')

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

        print(f'{s=}, {bench(per_token_quantize, x)=:.4f}, {bench(per_block_quantize, x)=:.4f}, {bench(per_token_cast_to_fp8, x)=:.4f}, {bench(per_block_cast_to_fp8, x)=:.4f}')
        
        


if __name__ == "__main__":

        
    # test_per_token_quantize()
    # test_per_block_quantize()
    benchmark()
    