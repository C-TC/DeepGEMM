import torch

from . import jit
from .jit_kernels import (
    gemm_fp8_fp8_bf16_nt,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    ceil_div,
    set_num_sms, get_num_sms,
    get_col_major_tma_aligned_tensor,
    get_m_alignment_for_contiguous_layout
)
from .utils import bench, bench_kineto, calc_diff
# TODO: replace this with fast implementation
from .triton_impl import per_token_quantize, transpose_per_token_quantize, per_block_quantize, transpose_per_block_quantize
