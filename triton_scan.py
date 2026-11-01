import triton
import triton.language as tl
import torch

@triton.jit
def selective_scan_fwd_kernel(
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
    delta_bias_ptr,
    stride_u_batch, stride_u_d, stride_u_l,
    stride_delta_batch, stride_delta_d, stride_delta_l,
    stride_A_d, stride_A_n,
    stride_B_batch, stride_B_n, stride_B_l,
    stride_C_batch, stride_C_n, stride_C_l,
    stride_out_batch, stride_out_d, stride_out_l,
    L: tl.constexpr,
    D_DIM: tl.constexpr,
    N_STATE: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DELTA_SOFTPLUS: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)

    offsets_n = tl.arange(0, N_STATE)
    a_ptrs = A_ptr + pid_d * stride_A_d + offsets_n * stride_A_n
    a_val = tl.load(a_ptrs)

    d_bias = 0.0
    if HAS_BIAS:
        d_bias = tl.load(delta_bias_ptr + pid_d)

    x = tl.zeros([N_STATE], dtype=tl.float32)

    u_base = u_ptr + pid_b * stride_u_batch + pid_d * stride_u_d
    delta_base = delta_ptr + pid_b * stride_delta_batch + pid_d * stride_delta_d
    out_base = out_ptr + pid_b * stride_out_batch + pid_d * stride_out_d
    z_base = z_ptr + pid_b * stride_u_batch + pid_d * stride_u_d

    for l in range(L):
        u_val = tl.load(u_base + l * stride_u_l)
        delta_val = tl.load(delta_base + l * stride_delta_l)
        if HAS_BIAS:
            delta_val = delta_val + d_bias
        if DELTA_SOFTPLUS:
            delta_val = tl.log(1.0 + tl.exp(delta_val))

        b_ptrs = B_ptr + pid_b * stride_B_batch + offsets_n * stride_B_n + l * stride_B_l
        b_val = tl.load(b_ptrs)

        c_ptrs = C_ptr + pid_b * stride_C_batch + offsets_n * stride_C_n + l * stride_C_l
        c_val = tl.load(c_ptrs)

        dA = tl.exp(delta_val * a_val)
        dB_u = (delta_val * u_val) * b_val

        x = dA * x + dB_u

        y_val = tl.sum(x * c_val)

        if HAS_D:
            d_val = tl.load(D_ptr + pid_d)
            y_val = y_val + d_val * u_val

        if HAS_Z:
            z_val = tl.load(z_base + l * stride_u_l)
            silu_z = z_val / (1.0 + tl.exp(-z_val))
            y_val = y_val * silu_z

        tl.store(out_base + l * stride_out_l, y_val)

def triton_selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    if D is not None: D = D.contiguous()
    if z is not None: z = z.contiguous()
    if delta_bias is not None: delta_bias = delta_bias.contiguous()

    B_sz, D_sz, L_sz = u.shape
    N_sz = A.shape[1]
    
    out = torch.empty_like(u)
    grid = (B_sz, D_sz)
    selective_scan_fwd_kernel[grid](
        u, delta, A, B, C, D, z, out, delta_bias,
        u.stride(0), u.stride(1), u.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        L=L_sz,
        D_DIM=D_sz,
        N_STATE=16,
        HAS_Z=(z is not None),
        HAS_D=(D is not None),
        HAS_BIAS=(delta_bias is not None),
        DELTA_SOFTPLUS=delta_softplus,
    )
    return out
