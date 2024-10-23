import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


# Level 1: Basic Self-Attention with Triton Kernel
@triton.jit
def attention_kernel(Q_ptr, K_ptr, V_ptr, out_ptr, scale, batch_size, seq_len, d_k, BLOCK_SIZE: tl.constexpr):
    # Program ID gives us batch and head indices
    pid = tl.program_id(0)
    batch_id = pid // seq_len
    token_id = pid % seq_len
    
    # Offsets to load Q, K, V
    q_offset = batch_id * seq_len * d_k + token_id * d_k
    k_offset = batch_id * seq_len * d_k
    v_offset = batch_id * seq_len * d_k
    
    # Load Q, K, V into local memory
    Q = tl.load(Q_ptr + q_offset + tl.arange(0, d_k))
    K = tl.load(K_ptr + k_offset + tl.arange(0, d_k))
    V = tl.load(V_ptr + v_offset + tl.arange(0, d_k))
    
    # Compute scores
    scores = tl.dot(Q, K.transpose(-1, -2)) * scale
    scores = tl.softmax(scores, dim=-1)
    
    # Compute final attention output
    out = tl.dot(scores, V)
    tl.store(out_ptr + q_offset, out)


class SelfAttHeadLevel1(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.WQ = nn.Linear(emb_dim, d_k)
        self.WK = nn.Linear(emb_dim, d_k)
        self.WV = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, E, mask=None):
        Q = self.WQ(E)
        K = self.WK(E)
        V = self.WV(E)

        out = torch.empty_like(Q)  # Allocate output tensor
        
        # Triton kernel launch
        attention_kernel[(Q.shape[0] * Q.shape[1])](
            Q, K, V, out, 1.0 / math.sqrt(self.d_k),
            Q.shape[0], Q.shape[1], self.d_k,
            BLOCK_SIZE=16
        )
        
        return out


# Level 2: Optimized with Bias Removal and Scaling
class SelfAttHeadLevel2(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.WQ = nn.Linear(emb_dim, d_k, bias=False)
        self.WK = nn.Linear(emb_dim, d_k, bias=False)
        self.WV = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = 1.0 / math.sqrt(d_k)
    
    def forward(self, E, mask=None):
        Q = self.WQ(E)
        K = self.WK(E)
        V = self.WV(E)

        out = torch.empty_like(Q)
        attention_kernel[(Q.shape[0] * Q.shape[1])](
            Q, K, V, out, self.scale,
            Q.shape[0], Q.shape[1], self.d_k,
            BLOCK_SIZE=16
        )
        
        return out


# Level 3: Combined Q and K Projection
class SelfAttHeadLevel3(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.WQK = nn.Linear(emb_dim, 2 * d_k, bias=False)
        self.WV = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, E, mask=None):
        QK = self.WQK(E)
        Q, K = torch.chunk(QK, 2, dim=-1)
        V = self.WV(E)

        out = torch.empty_like(Q)
        attention_kernel[(Q.shape[0] * Q.shape[1])](
            Q, K, V, out, self.scale,
            Q.shape[0], Q.shape[1], self.d_k,
            BLOCK_SIZE=16
        )
        
        return out


# Level 4: Single Matrix Projection with KV Cache
class SelfAttHeadLevel4(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.WQKV = nn.Linear(emb_dim, d_k * 2 + emb_dim, bias=False)
        self.scale = 1.0 / math.sqrt(d_k)
    
    def forward(self, E, mask=None, past_kv: Optional[KVCache] = None, use_cache: bool = False):
        qkv = self.WQKV(E)
        Q, K_new, V_new = torch.split(qkv, [self.d_k, self.d_k, self.emb_dim], dim=-1)

        if past_kv is not None:
            K = torch.cat([past_kv.key, K_new], dim=1)
            V = torch.cat([past_kv.value, V_new], dim=1)
            new_kv = KVCache(key=K, value=V)
        else:
            K, V = K_new, V_new
            new_kv = KVCache(key=K, value=V) if use_cache else None

        out = torch.empty_like(Q)
        attention_kernel[(Q.shape[0] * Q.shape[1])](
            Q, K, V, out, self.scale,
            Q.shape[0], Q.shape[1], self.d_k,
            BLOCK_SIZE=16
        )
        
        return out, new_kv


# Level 5: Maximum Efficiency with Shared Weights and Parameter Initialization
class SelfAttHeadLevel5(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        
        self.W = nn.Parameter(torch.empty(emb_dim, d_k * 2 + emb_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.scale = 1.0 / math.sqrt(d_k)
    
    def forward(self, E, mask=None, past_kv: Optional[KVCache] = None, use_cache: bool = False):
        combined = F.linear(E, self.W)
        Q, K_new, V_new = torch.split(combined, [self.d_k, self.d_k, self.emb_dim], dim=-1)

        if past_kv is not None:
            K = torch.cat([past_kv.key, K_new], dim=1).contiguous()
            V = torch.cat([past_kv.value, V_new], dim=1).contiguous()
            new_kv = KVCache(key=K, value=V)
        else:
            K, V = K_new.contiguous(), V_new.contiguous()
            new_kv = KVCache(key=K, value=V) if use_cache else None

        out = torch.empty_like(Q)
        attention_kernel[(Q.shape[0] * Q.shape[1])](
            Q, K, V, out, self.scale,
            Q.shape[0], Q.shape[1], self.d_k,
            BLOCK_SIZE=16
        )
        
        return out, new_kv


# Testing the models with dummy data
batch_size = 2
seq_len = 5
emb_dim = 8
d_k = 4

E = torch.randn(batch_size, seq_len, emb_dim)
mask = torch.ones(batch_size, seq_len).bool()

# Instantiate models and run forward pass
level1 = SelfAttHeadLevel1(emb_dim, d_k)
output1 = level1(E)

level2 = SelfAttHeadLevel2(emb_dim, d_k)
output2 = level2(E)

level3 = SelfAttHeadLevel3(emb_dim, d_k)
output3 = level3(E)

level4 = SelfAttHeadLevel4(emb_dim, d_k)
output4, _ = level4(E)

level5 = SelfAttHeadLevel5(emb_dim, d_k)
output5, _ = level5(E)

# Verify that all models produce consistent outputs
assert torch.allclose(output1, output2, atol=1e-5)
assert torch.allclose(output2, output3, atol=1e-5)
assert torch.allclose(output3, output4, atol=1e-5)
assert torch.allclose(output4, output5, atol=1e-5)

print("All models produce consistent results.")
