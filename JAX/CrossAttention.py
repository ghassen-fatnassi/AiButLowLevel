import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math
from functools import partial

@dataclass
class KVCache:
    key: jnp.ndarray
    value: jnp.ndarray

# Level 1: Basic Implementation
class SelfAttHeadLevel1(nn.Module):
    emb_dim: int
    d_k: int

    @nn.compact
    def __call__(self, E, mask=None):
        WQ = nn.Dense(features=self.d_k, name='WQ')
        WK = nn.Dense(features=self.d_k, name='WK')
        WV = nn.Dense(features=self.emb_dim, name='WV')

        Q = WQ(E)
        K = WK(E)
        V = WV(E)

        scores = jnp.matmul(Q, K.transpose(0, 2, 1))
        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = jnp.where(mask == 0, float('-inf'), scores)

        attention = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attention, V)
        
        return output

# Level 2: Optimized Scaling and Bias Removal
class SelfAttHeadLevel2(nn.Module):
    emb_dim: int
    d_k: int

    def setup(self):
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.WQ = nn.Dense(features=self.d_k, use_bias=False, name='WQ')
        self.WK = nn.Dense(features=self.d_k, use_bias=False, name='WK')
        self.WV = nn.Dense(features=self.emb_dim, use_bias=False, name='WV')

    def __call__(self, E, mask=None):
        Q = self.WQ(E)
        K = self.WK(E)
        V = self.WV(E)

        scores = jnp.einsum('bqd,bkd->bqk', Q, K)
        scores = scores * self.scale

        if mask is not None:
            scores = jnp.where(mask == 0, float('-inf'), scores)

        attention = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bqk,bkd->bqd', attention, V)
        
        return output

# Level 3: Combined QK Projections
class SelfAttHeadLevel3(nn.Module):
    emb_dim: int
    d_k: int

    def setup(self):
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.WQK = nn.Dense(features=2 * self.d_k, use_bias=False, name='WQK')
        self.WV = nn.Dense(features=self.emb_dim, use_bias=False, name='WV')

    def __call__(self, E, mask=None):
        QK = self.WQK(E)
        Q, K = jnp.split(QK, 2, axis=-1)
        V = self.WV(E)

        scores = jnp.einsum('bqd,bkd->bqk', Q, K) * self.scale

        if mask is not None:
            scores = jnp.where(mask == 0, float('-inf'), scores)

        attention = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bqk,bkd->bqd', attention, V)
        
        return output

# Level 4: Single Matrix Projection with KV Cache
class SelfAttHeadLevel4(nn.Module):
    emb_dim: int
    d_k: int

    def setup(self):
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.WQKV = nn.Dense(features=self.d_k * 2 + self.emb_dim, use_bias=False, name='WQKV')

    def __call__(self, E, mask=None, past_kv: Optional[KVCache] = None, use_cache: bool = False):
        qkv = self.WQKV(E)
        Q, K_new, V_new = jnp.split(qkv, [self.d_k, self.d_k * 2], axis=-1)
        V_new = V_new  # Last split

        if past_kv is not None:
            K = jnp.concatenate([past_kv.key, K_new], axis=1)
            V = jnp.concatenate([past_kv.value, V_new], axis=1)
            new_kv = KVCache(key=K, value=V)
        else:
            K, V = K_new, V_new
            new_kv = KVCache(key=K, value=V) if use_cache else None

        scores = jnp.einsum('bqd,bkd->bqk', Q, K) * self.scale

        if mask is not None:
            if past_kv is not None:
                full_seq_len = K.shape[1]
                mask = jnp.pad(mask, ((0, 0), (0, full_seq_len - mask.shape[-1])), constant_values=True)
            scores = jnp.where(mask == 0, float('-inf'), scores)

        attention = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bqk,bkd->bqd', attention, V)
        
        return output, new_kv

# Level 5: Maximum Efficiency with Parameter Initialization
class SelfAttHeadLevel5(nn.Module):
    emb_dim: int
    d_k: int

    def setup(self):
        self.scale = 1.0 / math.sqrt(self.d_k)
        kernel_init = nn.initializers.variance_scaling(
            scale=2.0, mode='fan_in', distribution='uniform'
        )
        self.W = self.param('W', kernel_init, (self.emb_dim, self.d_k * 2 + self.emb_dim))

    def __call__(
        self, 
        E: jnp.ndarray, 
        mask: Optional[jnp.ndarray] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        combined = jnp.einsum('...d,dk->...k', E, self.W)
        Q, K_new, V_new = jnp.split(combined, [self.d_k, self.d_k * 2], axis=-1)
        V_new = V_new  # Last split

        if past_kv is not None:
            K = jnp.concatenate([past_kv.key, K_new], axis=1)
            V = jnp.concatenate([past_kv.value, V_new], axis=1)
            new_kv = KVCache(key=K, value=V)
        else:
            K, V = K_new, V_new
            new_kv = KVCache(key=K, value=V) if use_cache else None

        scores = jnp.einsum('bqd,bkd->bqk', Q, K) * self.scale

        if mask is not None:
            if past_kv is not None:
                full_seq_len = K.shape[1]
                mask = jnp.pad(mask, ((0, 0), (0, full_seq_len - mask.shape[-1])), constant_values=True)
            scores = jnp.where(mask == 0, float('-inf'), scores)

        attention = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bqk,bkd->bqd', attention, V)
        
        return output, new_kv

# Test configuration and benchmarking code
def create_test_inputs(batch_size=2, seq_len=5, emb_dim=8):
    key = jax.random.PRNGKey(0)
    E = jax.random.normal(key, (batch_size, seq_len, emb_dim))
    mask = jnp.ones((batch_size, seq_len), dtype=bool)
    mask = mask.at[:, -2:].set(False)  # Mask the last two positions
    return E, mask

def test_and_benchmark(model_cls, model_name, batch_size=2, seq_len=5, emb_dim=8, d_k=4):
    print(f"\nTesting {model_name}...")
    
    # Initialize model
    model = model_cls(emb_dim=emb_dim, d_k=d_k)
    key = jax.random.PRNGKey(0)
    E, mask = create_test_inputs(batch_size, seq_len, emb_dim)
    
    # Initialize parameters
    variables = model.init(key, E, mask)
    
    # JIT compile the forward pass
    @jax.jit
    def forward(params, E, mask):
        return model.apply({"params": params}, E, mask)
    
    # Warm-up run
    params = variables["params"]
    output = forward(params, E, mask)
    
    print("Output shape:", output.shape if isinstance(output, jnp.ndarray) else output[0].shape)
    
    return output

if __name__ == "__main__":
    # Run tests for each model
    outputs = {}
    model_classes = [
        (SelfAttHeadLevel1, "SelfAttHeadLevel1"),
        (SelfAttHeadLevel2, "SelfAttHeadLevel2"),
        (SelfAttHeadLevel3, "SelfAttHeadLevel3"),
        (SelfAttHeadLevel4, "SelfAttHeadLevel4"),
        (SelfAttHeadLevel5, "SelfAttHeadLevel5")
    ]
    
    for model_cls, name in model_classes:
        outputs[name] = test_and_benchmark(model_cls, name)