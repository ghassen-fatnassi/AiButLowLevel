import jax.numpy as jnp
from jax import random
from jax import jit
import time
import jax
@jit
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(10.0)
print(selu(x))


key = random.key(1701)
x = random.normal(key, (1_000_000,))
start = time.time()

selu(x).block_until_ready()

end=time.time()
print(end-start)

selu_jitted=jit(selu)

start = time.time()
_= selu_jitted(x).block_until_ready()
end=time.time()
print(end-start)

start = time.time()
_= selu_jitted(x).block_until_ready()
end=time.time()
print(end-start)

print(jax.make_jaxpr(selu)(3.0))
print(jax.make_jaxpr(selu_jitted)(3.0))


