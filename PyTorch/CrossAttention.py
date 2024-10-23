import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import time


# Level 1: Basic Implementation
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

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        return output

# Level 2: Optimized Scaling and Bias Removal
class SelfAttHeadLevel2(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.register_buffer('scale', torch.tensor(d_k).sqrt().reciprocal())
        self.WQ = nn.Linear(emb_dim, d_k, bias=False)
        self.WK = nn.Linear(emb_dim, d_k, bias=False)
        self.WV = nn.Linear(emb_dim, emb_dim, bias=False)
        
    def forward(self, E, mask=None):
        Q = self.WQ(E)
        K = self.WK(E)
        V = self.WV(E)

        scores = torch.bmm(Q, K.transpose(1, 2))
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        output = torch.bmm(attention, V)
        
        return output

# Level 3: Combined QK Projections
class SelfAttHeadLevel3(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.WQK = nn.Linear(emb_dim, 2 * d_k, bias=False)
        self.WV = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer('scale', torch.tensor(d_k).sqrt().reciprocal())
        
    def forward(self, E, mask=None):
        QK = self.WQK(E)
        Q, K = QK.chunk(2, dim=-1)
        V = self.WV(E)

        scores = torch.einsum('bqd,bkd->bqk', Q, K) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        output = torch.einsum('bqk,bkd->bqd', attention, V)
        
        return output

@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


# Level 4: Single Matrix Projection with KV Cache
class SelfAttHeadLevel4(nn.Module):
    def __init__(self, emb_dim, d_k):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        self.WQKV = nn.Linear(emb_dim, d_k * 2 + emb_dim, bias=False)
        self.register_buffer('scale', torch.tensor(d_k).sqrt().reciprocal())
    
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

        scores = torch.einsum('bqd,bkd->bqk', Q, K) * self.scale

        if mask is not None:
            if past_kv is not None:
                full_seq_len = K.size(1)
                mask = F.pad(mask, (0, full_seq_len - mask.size(-1)), value=True)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        output = torch.einsum('bqk,bkd->bqd', attention, V)
        
        return output, new_kv

# Level 5: Maximum Efficiency with Parameter Initialization
class SelfAttHeadLevel5(nn.Module):
    def __init__(self, emb_dim: int, d_k: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.d_k = d_k
        
        self.W = nn.Parameter(torch.empty(emb_dim, d_k * 2 + emb_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.register_buffer('scale', torch.tensor(d_k).sqrt().reciprocal())
        
    def forward(
        self, 
        E: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        combined = F.linear(E, self.W)
        Q, K_new, V_new = torch.split(combined, [self.d_k, self.d_k, self.emb_dim], dim=-1)

        if past_kv is not None:
            K = torch.cat([past_kv.key, K_new], dim=1).contiguous()
            V = torch.cat([past_kv.value, V_new], dim=1).contiguous()
            new_kv = KVCache(key=K, value=V)
        else:
            K, V = K_new.contiguous(), V_new.contiguous()
            new_kv = KVCache(key=K, value=V) if use_cache else None

        scores = torch.einsum('bqd,bkd->bqk', Q, K) * self.scale

        if mask is not None:
            if past_kv is not None:
                full_seq_len = K.size(1)
                mask = F.pad(mask, (0, full_seq_len - mask.size(-1)), value=True)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1, dtype=torch.float32)
        output = torch.einsum('bqk,bkd->bqd', attention.to(V.dtype), V)
        
        return output, new_kv
    
# Test configuration
batch_size = 2
seq_len = 5
emb_dim = 8
d_k = 4
num_iterations = 100  # Number of iterations for benchmarking

# Sample input tensor and mask
E = torch.randn(batch_size, seq_len, emb_dim)
mask = torch.ones(batch_size, seq_len).bool()
mask[:, -2:] = 0  # Mask the last two positions

# Function to run tests, benchmarks, and output verification on each model
def test_and_benchmark(model, model_name):
    print(f"\nTesting {model_name}...")
    model.eval()  # Set the model to evaluation mode
    
    # Test output shape and optional features
    with torch.no_grad():
        if model_name in ["SelfAttHeadLevel4", "SelfAttHeadLevel5"]:
            output, new_kv = model(E, mask=mask, use_cache=True)
            print("Output shape:", output.shape)
            print("New KV Cache - Key shape:", new_kv.key.shape, "Value shape:", new_kv.value.shape)
        else:
            output = model(E, mask=mask)
            print("Output shape:", output.shape)
    
    # Benchmark the runtime
    start_time = time.time()
    for _ in range(num_iterations):
        if model_name in ["SelfAttHeadLevel4", "SelfAttHeadLevel5"]:
            _ = model(E, mask=mask, use_cache=True)
        else:
            _ = model(E, mask=mask)
    end_time = time.time()
    
    avg_time_per_run = (end_time - start_time) / num_iterations
    print(f"Average runtime per run: {avg_time_per_run:.6f} seconds")

    return output

# Instantiate each model and run tests
outputs = {}

level1 = SelfAttHeadLevel1(emb_dim, d_k)
outputs["SelfAttHeadLevel1"] = test_and_benchmark(level1, "SelfAttHeadLevel1")

level2 = SelfAttHeadLevel2(emb_dim, d_k)
outputs["SelfAttHeadLevel2"] = test_and_benchmark(level2, "SelfAttHeadLevel2")

level3 = SelfAttHeadLevel3(emb_dim, d_k)
outputs["SelfAttHeadLevel3"] = test_and_benchmark(level3, "SelfAttHeadLevel3")

level4 = SelfAttHeadLevel4(emb_dim, d_k)
outputs["SelfAttHeadLevel4"] = test_and_benchmark(level4, "SelfAttHeadLevel4")

level5 = SelfAttHeadLevel5(emb_dim, d_k)
outputs["SelfAttHeadLevel5"] = test_and_benchmark(level5, "SelfAttHeadLevel5")

# Verify that all models produce the same outputs
print("\nVerifying outputs consistency across models...")
tolerance = 1e-5
reference_output = outputs["SelfAttHeadLevel1"]

consistent = True
for name, output in outputs.items():
    if not torch.allclose(reference_output, output, atol=tolerance):
        consistent = False
        print(f"Output mismatch detected in {name}.")
    else:
        print(f"{name} matches the reference output.")

if consistent:
    print("\nAll models produce consistent outputs within the tolerance level.")
else:
    print("\nThere are inconsistencies in the outputs.")
