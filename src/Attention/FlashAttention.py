import torch
import triton
import triton.language as tl

@triton.jit
def _optimized_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    batch_idx = pid_batch_head // tl.num_programs(2)
    head_idx = pid_batch_head % tl.num_programs(2)
    batch_head_offset = batch_idx * stride_qb + head_idx * stride_qh
    Q_batch_head_ptr = Q_ptr + batch_head_offset
    K_batch_head_ptr = K_ptr + batch_head_offset
    V_batch_head_ptr = V_ptr + batch_head_offset
    O_batch_head_ptr = O_ptr + batch_head_offset
    seq_offset_m = pid_m * BLOCK_M
    offs_m = seq_offset_m + tl.arange(0, BLOCK_M)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    Q_block_ptr = Q_batch_head_ptr + seq_offset_m * stride_qs
    q = tl.load(Q_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qs + 
                tl.arange(0, head_dim)[None, :] * stride_qd)
    for seq_offset_n in range(0, seq_len, BLOCK_N):
        offs_n = seq_offset_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_batch_head_ptr + offs_n[:, None] * stride_ks + 
                   tl.arange(0, head_dim)[None, :] * stride_kd)
        v = tl.load(V_batch_head_ptr + offs_n[:, None] * stride_vs + 
                   tl.arange(0, head_dim)[None, :] * stride_vd)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        if CAUSAL:
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float('-inf'))
        m_prev = m_i
        m_i = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_prev - m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_i[:, None]), 1)
        p = tl.exp(qk - m_i[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v)
    acc = acc / l_i[:, None]
    tl.store(O_batch_head_ptr + offs_m[:, None] * stride_os + 
             tl.arange(0, head_dim)[None, :] * stride_od, acc)
    M_ptr_idx = pid_batch_head * seq_len + offs_m
    tl.store(M_ptr + M_ptr_idx, m_i + tl.log(l_i))

@triton.jit
def _optimized_attn_bwd_dqkv(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr, M_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    seq_len, head_dim, num_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, MODE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads
    batch_head_offset = batch_idx * stride_qb + head_idx * stride_qh
    
    if MODE == 0:
        block_idx = pid
        offset_n = block_idx * BLOCK_N
        k_ptrs = K_ptr + batch_head_offset + offset_n * stride_ks
        v_ptrs = V_ptr + batch_head_offset + offset_n * stride_vs
        k_block = tl.load(k_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_ks + 
                          tl.arange(0, head_dim)[None, :] * stride_kd)
        v_block = tl.load(v_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_vs + 
                          tl.arange(0, head_dim)[None, :] * stride_vd)
        for offset_m in range(0, seq_len, BLOCK_M):
            offs_m = offset_m + tl.arange(0, BLOCK_M)
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            q_ptrs = Q_ptr + batch_head_offset + offset_m * stride_qs
            dq_ptrs = dQ_ptr + batch_head_offset + offset_m * stride_qs
            q_block = tl.load(q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qs + 
                              tl.arange(0, head_dim)[None, :] * stride_qd)
            dq_block = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
            do_ptrs = dO_ptr + batch_head_offset + offset_m * stride_dos
            do_block = tl.load(do_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_dos + 
                               tl.arange(0, head_dim)[None, :] * stride_dod)
            m_ptrs = M_ptr + batch_head_idx * seq_len + offs_m
            m_block = tl.load(m_ptrs)
            qk = tl.dot(q_block, tl.trans(k_block)) * softmax_scale
            if CAUSAL:
                mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(mask, qk, float('-inf'))
            p = tl.exp(qk - m_block[:, None])
            dp = tl.dot(do_block, tl.trans(v_block))
            delta = tl.sum(do_block * q_block, axis=1)
            ds = p * (dp - delta[:, None])
            dq_block += softmax_scale * tl.dot(ds, k_block)
            tl.store(dq_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_dqs + 
                     tl.arange(0, head_dim)[None, :] * stride_dqd, dq_block)
    else:
        block_idx = pid
        offset_m = block_idx * BLOCK_M
        q_ptrs = Q_ptr + batch_head_offset + offset_m * stride_qs
        q_block = tl.load(q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qs + 
                          tl.arange(0, head_dim)[None, :] * stride_qd)
        do_ptrs = dO_ptr + batch_head_offset + offset_m * stride_dos
        do_block = tl.load(do_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_dos + 
                           tl.arange(0, head_dim)[None, :] * stride_dod)
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        m_ptrs = M_ptr + batch_head_idx * seq_len + offs_m
        m_block = tl.load(m_ptrs)
        for offset_n in range(0, seq_len, BLOCK_N):
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            k_ptrs = K_ptr + batch_head_offset + offset_n * stride_ks
            v_ptrs = V_ptr + batch_head_offset + offset_n * stride_vs
            dk_ptrs = dK_ptr + batch_head_offset + offset_n * stride_dks
            dv_ptrs = dV_ptr + batch_head_offset + offset_n * stride_dvs
            k_block = tl.load(k_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_ks + 
                              tl.arange(0, head_dim)[None, :] * stride_kd)
            v_block = tl.load(v_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_vs + 
                              tl.arange(0, head_dim)[None, :] * stride_vd)
            dk_block = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
            dv_block = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
            qk = tl.dot(q_block, tl.trans(k_block)) * softmax_scale
            if CAUSAL:
                mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(mask, qk, float('-inf'))
            p = tl.exp(qk - m_block[:, None])
            dv_block += tl.dot(tl.trans(p), do_block)
            delta = tl.sum(do_block * q_block, axis=1)
            dp = tl.dot(do_block, tl.trans(v_block))
            ds = p * (dp - delta[:, None])
            dk_block += softmax_scale * tl.dot(tl.trans(ds), q_block)
            tl.store(dk_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_dks + 
                     tl.arange(0, head_dim)[None, :] * stride_dkd, dk_block)
            tl.store(dv_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_dvs + 
                     tl.arange(0, head_dim)[None, :] * stride_dvd, dv_block)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=2),
    ],
    key=['seq_len', 'head_dim'],
)
def attention_forward(q, k, v, causal=False):
    batch_size, num_heads, seq_len, head_dim = q.shape
    assert k.shape[:3] == v.shape[:3] == (batch_size, num_heads, seq_len)
    assert k.shape[3] == v.shape[3] == head_dim
    o = torch.empty_like(q)
    m = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    BLOCK_M = min(128, triton.next_power_of_2(seq_len))
    BLOCK_N = min(64, triton.next_power_of_2(head_dim))
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    _optimized_attn_fwd_kernel[grid](
        Q_ptr=q, K_ptr=k, V_ptr=v, O_ptr=o, M_ptr=m,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qs=q.stride(2), stride_qd=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_ks=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vs=v.stride(2), stride_vd=v.stride(3),
        stride_ob=o.stride(0), stride_oh=o.stride(1), stride_os=o.stride(2), stride_od=o.stride(3),
        seq_len=seq_len, head_dim=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal
    )
    return o, m

def attention_backward(q, k, v, do, m, causal=False):
    batch_size, num_heads, seq_len, head_dim = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    BLOCK_M = min(128, triton.next_power_of_2(seq_len))
    BLOCK_N = min(64, triton.next_power_of_2(head_dim))
    grid = (triton.cdiv(seq_len, BLOCK_N), batch_size * num_heads)
    _optimized_attn_bwd_dqkv[grid](
        Q_ptr=q, K_ptr=k, V_ptr=v, dO_ptr=do, dQ_ptr=dq, dK_ptr=dk, dV_ptr=dv, M_ptr=m,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qs=q.stride(2), stride_qd=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_ks=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vs=v.stride(2), stride_vd=v.stride(3),
        stride_dob=do.stride(0), stride_doh=do.stride(1), stride_dos=do.stride(2), stride_dod=do.stride(3),
        stride_dqb=dq.stride(0), stride_dqh=dq.stride(1), stride_dqs=dq.stride(2), stride_dqd=dq.stride(3),
        stride_dkb=dk.stride(0), stride_dkh=dk.stride(1), stride_dks=dk.stride(2), stride_dkd=dk.stride(3),
        stride_dvb=dv.stride(0), stride_dvh=dv.stride(1), stride_dvs=dv.stride(2), stride_dvd=dv.stride(3),
        seq_len=seq_len, head_dim=head_dim, num_heads=num_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal, MODE=0
    )
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    _optimized_attn_bwd_dqkv[grid](
        Q_ptr=q, K_ptr=k, V_ptr=v, dO_ptr=do, dQ_ptr=dq, dK_ptr=dk, dV_ptr=dv, M_ptr=m,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qs=q.stride(2), stride_qd=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_ks=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vs=v.stride(2), stride_vd=v.stride(3),
        stride_dob=do.stride(0), stride_doh=do.stride(1), stride_dos=do.stride(2), stride_dod=do.stride(3),
        stride_dqb=dq.stride(0), stride_dqh=dq.stride(1), stride_dqs=dq.stride(2), stride_dqd=dq.stride(3),
        stride_dkb=dk.stride(0), stride_dkh=dk.stride(1), stride_dks=dk.stride(2), stride_dkd=dk.stride(3),
        stride_dvb=dv.stride(0), stride_dvh=dv.stride(1), stride_dvs=dv.stride(2), stride_dvd=dv.stride(3),
        seq_len=seq_len, head_dim=head_dim, num_heads=num_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal, MODE=1
    )
    return dq, dk, dv

class OptimizedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        q, k, v = map(lambda x: x.contiguous(), (q, k, v))
        o, m = attention_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, m)
        ctx.causal = causal
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, m = ctx.saved_tensors
        causal = ctx.causal
        do = do.contiguous()
        dq, dk, dv = attention_backward(q, k, v, do, m, causal)
        return dq, dk, dv, None

def optimized_attention(q, k, v, causal=False):
    return OptimizedAttention.apply(q, k, v, causal)

def test_optimized_attention(batch_size=8, num_heads=16, seq_len=4096, head_dim=64, causal=True, dtype=torch.float16):
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").requires_grad_()
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").requires_grad_()
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").requires_grad_()
    q1, k1, v1 = q.clone().detach().requires_grad_(), k.clone().detach().requires_grad_(), v.clone().detach().requires_grad_()
    softmax_scale = 1.0 / (head_dim ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), diagonal=1).bool()
        attn_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    ref_out = torch.matmul(attn_probs, v)
    opt_out = optimized_attention(q1, k1, v1, causal)
    torch.testing.assert_close(ref_out, opt_out, rtol=1e-3, atol=1e-3)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    opt_out.backward(grad_out.clone())
    torch.testing.assert_close(q.grad, q1.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k.grad, k1.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v.grad, v1.grad, rtol=1e-3, atol=1e-3)
    print("All tests passed!")

if __name__ == "__main__":
    test_optimized_attention(batch_size=8, num_heads=16, seq_len=4096, head_dim=64, causal=True)
    test_optimized_attention(batch_size=8, num_heads=16, seq_len=4096, head_dim=64, causal=False)
    print("PERFORMANCE OPTIMIZED ATTENTION TESTS PASSED")