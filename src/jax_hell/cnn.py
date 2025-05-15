import jax.numpy as jnp
from jax import grad, jit, vmap, random, make_jaxpr
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
import numpy as np
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST
import time

# --- Init & Utilities ---
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def relu(x): return jnp.maximum(0, x)

def debug_layer(w, x, b, name="layer"):
    out = jnp.dot(w, x) + b
    print(f"[{name}] w: {w.shape}, x: {x.shape}, b: {b.shape}")
    print(f"[{name}] out dtype: {out.dtype}, out shape: {out.shape}")
    return out

def predict_debug(params, image):
    activations = image
    for i, (w, b) in enumerate(params[:-1]):
        activations = relu(debug_layer(w, activations, b, f"Layer {i}"))
    final_w, final_b = params[-1]
    logits = debug_layer(final_w, activations, final_b, "Final Layer")
    return logits - logsumexp(logits)

batched_predict_debug = vmap(predict_debug, in_axes=(None, 0))

# --- Original predict ---
@jit
def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        activations = relu(jnp.dot(w, activations) + b)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

# --- Loss / Accuracy ---
@jit
def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)
@jit
def accuracy(params, images, targets):
    pred_class = jnp.argmax(batched_predict(params, images), axis=1)
    true_class = jnp.argmax(targets, axis=1)
    return jnp.mean(pred_class == true_class)
@jit
def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

def inspect_grads(params, x, y):
    grads = grad(loss)(params, x, y)
    for i, (dw, db) in enumerate(grads):
        print(f"Grad Layer {i}: dw {dw.shape} {dw.dtype}, db {db.shape} {db.dtype}")

# --- Dataset ---
@jit
def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))
@jit
def flatten_and_cast(pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

# --- Config ---
layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 8
batch_size = 1024
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))

# --- Load data ---
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)
# Get the full train dataset (for checking accuracy while training)
train_images = jnp.array(mnist_dataset.data.numpy().reshape(len(mnist_dataset.data), -1), dtype=jnp.float32)
train_labels = one_hot(np.asarray(mnist_dataset.targets), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), -1), dtype=jnp.float32)
test_labels = one_hot(np.asarray(mnist_dataset_test.targets), n_targets)


# --- Debug on first batch before training ---
print("\n--- DEBUG: Forward pass on first batch ---")
x0, y0 = next(iter(training_generator))
y0 = one_hot(y0, n_targets)
batched_predict_debug(params, x0)

print("\n--- DEBUG: JAXPR ---")
print(make_jaxpr(batched_predict)(params, x0))

print("\n--- DEBUG: JIT timing ---")
batched_predict_jit = jit(batched_predict)
t0 = time.time(); batched_predict_jit(params, x0); print("JIT first call:", time.time() - t0)
t0 = time.time(); batched_predict_jit(params, x0); print("JIT second call:", time.time() - t0)

print("\n--- DEBUG: HLO IR ---")
print(batched_predict_jit.lower(params, x0).as_text())

print("\n--- DEBUG: Gradients ---")
inspect_grads(params, x0, y0)

# --- Training loop ---
for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in training_generator:
        y = one_hot(y, n_targets)
        params = update(params, x, y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
