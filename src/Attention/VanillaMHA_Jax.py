import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, value_and_grad
import math
import optax # For optimizer
from flax import struct # For dataclasses
from functools import partial # For JIT compilation with static args

# --- Configuration ---
@struct.dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    d_mlp: int
    num_heads: int
    num_blocks: int
    max_seq_len: int  # Max sequence length for positional encoding
    dropout_rate: float = 0.1
    # This internal attention temperature is non-standard.
    # If not desired, set to 1.0. It's different from sampling_temperature.
    attention_temperature: float = 1.0

    @property
    def d_head(self):
        """Dimension of each attention head."""
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return self.d_model // self.num_heads

# --- Initialization Functions ---

def init_embedding_params(key, config: TransformerConfig):
    """Initialize embedding parameters."""
    key, subkey = random.split(key)
    embedding = random.normal(subkey, (config.vocab_size, config.d_model)) * 0.02
    return embedding

def init_linear_params(key, in_dim, out_dim, bias=True, init_scale_factor=2.0):
    """Initialize parameters for a linear layer with Kaiming He normal initialization."""
    key_w, key_b = random.split(key)
    # Kaiming He initialization for weights
    w = random.normal(key_w, (in_dim, out_dim)) * jnp.sqrt(init_scale_factor / in_dim)
    
    params = {'weight': w}
    if bias:
        b = jnp.zeros((out_dim,))
        params['bias'] = b
    return params

def init_layer_norm_params(config: TransformerConfig):
    """Initialize layer normalization parameters."""
    return {
        'gamma': jnp.ones((config.d_model,)),
        'beta': jnp.zeros((config.d_model,))
    }

def init_mlp_params(key, config: TransformerConfig):
    """Initialize MLP parameters."""
    key_l1, key_l2 = random.split(key)
    return {
        'l1': init_linear_params(key_l1, config.d_model, config.d_mlp),
        'l2': init_linear_params(key_l2, config.d_mlp, config.d_model)
    }

def init_attention_params(key, config: TransformerConfig):
    """Initialize multi-head attention parameters (simplified)."""
    keys = random.split(key, 4) # W_Q, W_K, W_V, W_O
    d_model = config.d_model
    return {
        'W_Q': init_linear_params(keys[0], d_model, d_model, bias=False),
        'W_K': init_linear_params(keys[1], d_model, d_model, bias=False),
        'W_V': init_linear_params(keys[2], d_model, d_model, bias=False),
        'W_O': init_linear_params(keys[3], d_model, d_model, bias=False)
    }

def init_decoder_block_params(key, config: TransformerConfig):
    """Initialize decoder block parameters."""
    key_att, key_mlp = random.split(key)
    # Note: LayerNorm params don't need separate keys as they are not random
    return {
        'attention': init_attention_params(key_att, config),
        'mlp': init_mlp_params(key_mlp, config),
        'ln1': init_layer_norm_params(config),
        'ln2': init_layer_norm_params(config)
    }

def init_decoder_params(key, config: TransformerConfig):
    """Initialize decoder parameters."""
    keys = random.split(key, config.num_blocks)
    blocks = [
        init_decoder_block_params(keys[i], config)
        for i in range(config.num_blocks)
    ]
    return {
        'blocks': blocks,
        'final_ln': init_layer_norm_params(config)
    }

def init_projection_params(key, config: TransformerConfig):
    """Initialize projection (to vocab) parameters."""
    return init_linear_params(key, config.d_model, config.vocab_size)

def init_transformer_params(key, config: TransformerConfig):
    """Initialize all transformer parameters."""
    keys = random.split(key, 3) # embedding, decoder, projection
    
    return {
        'embedding': init_embedding_params(keys[0], config),
        'decoder': init_decoder_params(keys[1], config),
        'projection': init_projection_params(keys[2], config)
    }

# --- Core Layer Functions ---

def linear(params, x):
    """Apply linear transformation."""
    y = x @ params['weight']
    if 'bias' in params:
        y = y + params['bias']
    return y

def layer_norm(params, x, eps=1e-5):
    """Apply layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return params['gamma'] * normalized + params['beta']

def mlp_block(params, x):
    """Apply MLP block with ReLU activation."""
    h = jax.nn.relu(linear(params['l1'], x))
    # Note: No dropout in MLP in the original code, kept as is.
    # If dropout is desired here, it should be added.
    return linear(params['l2'], h)

def attention(q, k, v, mask, temperature, dropout_rate, key, deterministic):
    """Compute attention scores and apply them to values."""
    # q, k, v: (batch, num_heads, seq_len, d_head)
    # mask: (batch, 1, seq_len_q, seq_len_k) or (1, 1, seq_len_q, seq_len_k)
    d_k = q.shape[-1]
    att_scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k) # (batch, num_heads, seq_len_q, seq_len_k)
    
    if mask is not None:
        # Mask is typically (1, 1, seq_len, seq_len) for causal or (batch, 1, 1, seq_len_k) for padding
        att_scores = jnp.where(mask == 0, -1e9, att_scores)
    
    # Non-standard: temperature scaling on attention scores
    att_scores = att_scores / temperature 
    att_weights = jax.nn.softmax(att_scores, axis=-1) # (batch, num_heads, seq_len_q, seq_len_k)
    
    if not deterministic and dropout_rate > 0:
        att_weights = jax.nn.dropout(key, rate=dropout_rate, x=att_weights, deterministic=deterministic)
            
    weighted_values = jnp.matmul(att_weights, v) # (batch, num_heads, seq_len_q, d_head)
    return weighted_values

def multi_head_attention(params, x, mask, config: TransformerConfig, key, deterministic: bool):
    """Apply multi-head attention mechanism."""
    batch_size, seq_len, d_model = x.shape
    num_heads = config.num_heads
    d_head = config.d_head # d_model // num_heads

    # Linear projections
    Q = linear(params['W_Q'], x)  # (batch, seq_len, d_model)
    K = linear(params['W_K'], x)  # (batch, seq_len, d_model)
    V = linear(params['W_V'], x)  # (batch, seq_len, d_model)
    
    # Reshape for multi-head: (batch, seq_len, num_heads, d_head)
    # Then transpose to: (batch, num_heads, seq_len, d_head)
    Q_headed = Q.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K_headed = K.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V_headed = V.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    
    # Apply attention mechanism
    # key for dropout within attention
    attn_dropout_key = None
    if key is not None: # key is only None if deterministic is True and no key is passed.
        key, attn_dropout_key = random.split(key)

    attn_output_headed = attention(
        Q_headed, K_headed, V_headed, mask,
        temperature=config.attention_temperature,
        dropout_rate=config.dropout_rate,
        key=attn_dropout_key,
        deterministic=deterministic
    ) # (batch, num_heads, seq_len, d_head)
    
    # Transpose back: (batch, seq_len, num_heads, d_head)
    # Then reshape to: (batch, seq_len, d_model)
    attn_output_concat = attn_output_headed.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Output projection
    return linear(params['W_O'], attn_output_concat)


# --- Model Component Functions ---

def embed_tokens(params, tokens):
    """Convert token indices to embeddings and scale them by sqrt(d_model)."""
    embedding_table = params['embedding']
    d_model = embedding_table.shape[1]
    return jnp.take(embedding_table, tokens, axis=0) * jnp.sqrt(d_model)

def create_positional_encoding(max_seq_len, d_model):
    """Create positional encodings."""
    PE = jnp.zeros((max_seq_len, d_model))
    pos = jnp.arange(0, max_seq_len)[:, None] # (max_seq_len, 1)
    # Original Transformer: div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    # User's version:
    in_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # Corrected 1e5 to 10000.0
    
    PE = PE.at[:, 0::2].set(jnp.sin(pos * in_term))
    PE = PE.at[:, 1::2].set(jnp.cos(pos * in_term))
    return PE[None, :, :]  # Add batch dimension: (1, max_seq_len, d_model)

def add_positional_encoding(embeddings, PE):
    """Add positional encoding to embeddings."""
    # embeddings: (batch, seq_len, d_model)
    # PE: (1, max_seq_len, d_model)
    seq_len = embeddings.shape[1]
    return embeddings + PE[:, :seq_len, :]

def decoder_block(params, x, mask, config: TransformerConfig, key, deterministic: bool):
    """Apply decoder block with self-attention and MLP."""
    # key is for dropout, split if necessary
    att_key, mlp_dropout_key = None, None # mlp_dropout_key unused unless MLP has dropout
    if key is not None and not deterministic:
        key, att_key, mlp_dropout_key = random.split(key, 3)
    
    # Self-attention with residual connection and layer norm
    ln1_out = layer_norm(params['ln1'], x)
    att_output = multi_head_attention(
        params['attention'], ln1_out, mask, config, att_key, deterministic
    )
    # Note: Original code does not have dropout after attention here, but inside MHA on weights.
    # If dropout on att_output itself is desired:
    # if not deterministic and config.dropout_rate > 0:
    #     att_output = jax.nn.dropout(some_key, rate=config.dropout_rate, x=att_output, deterministic=deterministic)
    x = x + att_output 
    
    # MLP with residual connection and layer norm
    ln2_out = layer_norm(params['ln2'], x)
    mlp_output = mlp_block(params['mlp'], ln2_out)
    # If dropout on mlp_output is desired:
    # if not deterministic and config.dropout_rate > 0:
    #    mlp_output = jax.nn.dropout(mlp_dropout_key, rate=config.dropout_rate, x=mlp_output, deterministic=deterministic)
    x = x + mlp_output
    
    return x

def decoder(params, x, mask, config: TransformerConfig, key, deterministic: bool):
    """Apply transformer decoder."""
    block_keys = None
    if key is not None and not deterministic and config.num_blocks > 0:
        block_keys = random.split(key, config.num_blocks)
    
    for i, block_params in enumerate(params['blocks']):
        current_block_key = block_keys[i] if block_keys is not None else None
        x = decoder_block(
            block_params, x, mask, config, current_block_key, deterministic
        )
    
    return layer_norm(params['final_ln'], x)

def create_causal_mask(seq_len):
    """Create causal mask for decoder's self-attention."""
    # (seq_len, seq_len)
    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=1)
    return (~mask).astype(jnp.float32) # True for attend (1.0), False for mask (0.0) -> then 0 becomes -1e9

def transformer_forward(params, tokens, pe, config: TransformerConfig, key, deterministic: bool):
    """Forward pass of the transformer model."""
    # key is for dropout, will be split and passed down.
    decoder_key = None
    if key is not None and not deterministic:
        key, decoder_key = random.split(key) # Ensure key is further splittable if needed
    
    # Embedding and positional encoding
    x = embed_tokens(params, tokens) # (batch, seq_len, d_model)
    x = add_positional_encoding(x, pe) # pe:(1, max_seq_len, d_model)
    
    # Dropout after embeddings + PE (common practice)
    if not deterministic and config.dropout_rate > 0:
        # Need a key for this dropout if key is not None
        emb_dropout_key = None
        if decoder_key is not None: # Reuse split logic, or split a new one from `key`
             decoder_key, emb_dropout_key = random.split(decoder_key) # Example of splitting
        elif key is not None: # if decoder_key was not created because num_blocks = 0
             key, emb_dropout_key = random.split(key)

        if emb_dropout_key is not None:
             x = jax.nn.dropout(emb_dropout_key, rate=config.dropout_rate, x=x, deterministic=deterministic)

    # Create causal mask based on sequence length
    seq_len = tokens.shape[1]
    # Causal mask (1, 1, seq_len, seq_len) for self-attention
    causal_mask = create_causal_mask(seq_len)[None, None, :, :]
    
    # Apply decoder
    decoder_output = decoder(
        params['decoder'], x, causal_mask, config, decoder_key, deterministic
    )
    
    # Project to vocabulary
    logits = linear(params['projection'], decoder_output) # (batch, seq_len, vocab_size)
    return logits

# --- Generation Function ---

def generate_tokens(
    params, input_ids, pe, config: TransformerConfig, max_new_tokens: int,
    sampling_temperature: float = 1.0, top_k: int = None, key=None
):
    """Generate tokens autoregressively."""
    # input_ids: (batch_size, current_seq_len)
    # pe: (1, max_config_seq_len, d_model)
    # For generation, dropout is off (deterministic=True)
    # The dropout_key for transformer_forward can be None if deterministic=True
    
    current_ids = input_ids
    
    for _ in range(max_new_tokens):
        # Ensure current_ids don't exceed max_seq_len for PE
        current_seq_len = current_ids.shape[1]
        if current_seq_len >= config.max_seq_len:
            print(f"Warning: Sequence length {current_seq_len} reached max_seq_len {config.max_seq_len} for PE during generation.")
            # Option: truncate current_ids if strict, or hope PE slicing handles it
            # For now, we assume pe is large enough or slicing handles it.
            # Transformer forward will use PE[:, :current_seq_len, :]

        # Key management for sampling (if needed for top_k/nucleus) and internal dropout (though deterministic)
        step_key = None
        if key is not None:
            key, step_key, sample_key = random.split(key, 3)
        else: # If main key is None, generate a temporary one if needed for sampling
            sample_key = random.PRNGKey(0) # Fallback, not ideal for repeated calls without external key

        # Forward pass - deterministic=True means dropout_key for transformer_forward can be None
        logits = transformer_forward(
            params, current_ids, pe, config, 
            key=None, # No dropout key needed as deterministic=True
            deterministic=True
        ) # (batch_size, current_seq_len, vocab_size)
        
        # Get logits for the next token
        next_token_logits = logits[:, -1, :] # (batch_size, vocab_size)
        
        # Apply sampling temperature
        next_token_logits = next_token_logits / sampling_temperature
        
        # Optional: Top-k sampling
        if top_k is not None and top_k > 0:
            top_k = min(top_k, next_token_logits.shape[-1]) # Ensure k is not larger than vocab size
            # Get top-k logits and their indices
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, k=top_k)
            # Create a mask for non-top-k logits
            mask = jnp.full_like(next_token_logits, -jnp.inf)
            # Fill in the top-k logits
            # This is a bit tricky with vmap or scatter, simpler to reconstruct if batch_size=1
            # For batch > 1:
            def set_top_k(logits_row, indices_row, values_row):
                return jnp.zeros_like(logits_row).at[indices_row].set(values_row)

            # This creates a sparse logit tensor for top_k, not ideal, better to sample from top_k_logits directly
            # A common way:
            # Set non-top-k logits to -inf
            kth_values = jnp.min(top_k_logits, axis=-1, keepdims=True) # Smallest of the top-k values
            next_token_logits = jnp.where(next_token_logits < kth_values, -jnp.inf, next_token_logits)


        # Sample next token
        # jax.random.categorical requires a key
        next_token = random.categorical(sample_key, next_token_logits, axis=-1) # (batch_size,)
        
        # Add the new token to our running sequence
        next_token = next_token[:, None]  # (batch_size, 1) to enable concatenation
        current_ids = jnp.concatenate([current_ids, next_token], axis=1)
    
    return current_ids

# --- Training Components ---

@partial(jit, static_argnums=(3,)) # JIT loss_fn, config is static
def loss_fn(params, batch_tokens, pe, config: TransformerConfig, dropout_key):
    """Calculates cross-entropy loss for language modeling."""
    # batch_tokens: (batch_size, seq_length)
    # Input for the model: all but the last token
    # Target for the model: all but the first token
    input_tokens = batch_tokens[:, :-1]
    target_tokens = batch_tokens[:, 1:]
    
    # Get logits from the model (training mode, so deterministic=False)
    # dropout_key is essential here
    logits = transformer_forward(params, input_tokens, pe, config, key=dropout_key, deterministic=False)
    # logits: (batch_size, seq_length-1, vocab_size)
    # target_tokens: (batch_size, seq_length-1)

    # Compute cross-entropy loss
    # optax.softmax_cross_entropy expects logits of shape (..., num_classes)
    # and labels of shape (...)
    loss_values = optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens)
    
    # Mask out padding tokens if any (not implemented here, assumes no padding)
    # For now, just mean loss
    return jnp.mean(loss_values)

@partial(jit, static_argnums=(3, 4)) # JIT train_step, config and optimizer are static
def train_step(params, opt_state, batch_tokens, config: TransformerConfig, optimizer: optax.GradientTransformation, pe, step_dropout_key):
    """Performs a single training step."""
    loss, grads = value_and_grad(loss_fn)(params, batch_tokens, pe, config, step_dropout_key)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# --- Example Usage: Generation and Training ---

def run_generation_example(config: TransformerConfig, params, pe, key_gen):
    """Demonstrates token generation."""
    print("\n--- Running Generation Example ---")
    batch_size = 2
    prompt_len = 5
    max_new_tokens_to_gen = 10
    
    # Create dummy initial tokens (prompt)
    key_prompt, key_gen_step = random.split(key_gen)
    prompt_tokens = random.randint(key_prompt, (batch_size, prompt_len), 0, config.vocab_size)
    
    print(f"Initial prompt tokens (shape {prompt_tokens.shape}):\n{prompt_tokens}")

    # JIT the generation function for this specific config and pe
    # Note: max_new_tokens and sampling_temperature are dynamic here.
    # If they were static, they could also be part of partial.
    # For top_k, if it's always None or always a specific value, it could be static.
    # Jit might be slow for the first call if shapes change often or control flow is complex inside.
    # For simpler cases, jit is fine.
    # @partial(jit, static_argnames=("config", "max_new_tokens", "top_k")) # Need to pass pe as static if jitted
    # def jitted_generate_tokens_func(...)
    # For now, not JITing generate_tokens here, but can be done.

    generated_ids = generate_tokens(
        params, prompt_tokens, pe, config,
        max_new_tokens=max_new_tokens_to_gen,
        sampling_temperature=0.8, # Example temperature
        top_k=5,                 # Example top-k
        key=key_gen_step
    )
    print(f"Generated tokens (shape {generated_ids.shape}):\n{generated_ids}")
    return generated_ids

def run_training_loop_example(config: TransformerConfig, params, pe, key_train_main):
    """Demonstrates a basic training loop."""
    print("\n--- Running Training Loop Example ---")
    num_epochs = 3 # Small number of epochs for demo
    batch_size_train = 4 # Small batch for demo
    steps_per_epoch = 10 # Small number of steps for demo
    learning_rate = 1e-4

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        key_epoch, key_train_main = random.split(key_train_main) # New key for each epoch's data/dropout

        for step in range(steps_per_epoch):
            # Generate a dummy batch of data
            # In a real scenario, you'd load data from a dataset
            key_batch, key_dropout_step, key_epoch = random.split(key_epoch, 3)
            dummy_batch_tokens = random.randint(
                key_batch,
                (batch_size_train, config.max_seq_len), # Use max_seq_len for fixed shape training
                0,
                config.vocab_size
            )
            
            params, opt_state, loss_value = train_step(
                params, opt_state, dummy_batch_tokens, config, optimizer, pe, key_dropout_step
            )
            epoch_loss += loss_value
            
            if (step + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss_value:.4f}")
        
        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
    
    print("Training finished.")
    return params # Return trained parameters


if __name__ == "__main__":
    # --- Define Model Configuration ---
    config = TransformerConfig(
        vocab_size=50,     # e.g., size of your vocabulary
        d_model=64,        # Embedding dimension and model dimension
        d_mlp=128,         # Dimension of the MLP's hidden layer
        num_heads=4,       # Number of attention heads (d_model must be divisible by num_heads)
        num_blocks=3,      # Number of decoder blocks
        max_seq_len=32,    # Maximum sequence length the model can handle (for PE)
        dropout_rate=0.1,
        attention_temperature=1.0 # Default, standard attention
    )
    print("Model Configuration:")
    print(f" vocab_size: {config.vocab_size}")
    print(f" d_model: {config.d_model}")
    print(f" d_mlp: {config.d_mlp}")
    print(f" num_heads: {config.num_heads}")
    print(f" d_head: {config.d_head}")
    print(f" num_blocks: {config.num_blocks}")
    print(f" max_seq_len: {config.max_seq_len}")
    print(f" dropout_rate: {config.dropout_rate}")
    print(f" attention_temperature (internal): {config.attention_temperature}")

    # --- Initialize Keys ---
    main_key = random.PRNGKey(42)
    key_init, key_gen, key_train = random.split(main_key, 3)

    # --- Initialize Model Parameters ---
    print("\nInitializing model parameters...")
    transformer_params = init_transformer_params(key_init, config)
    # print("Sample of parameters (embedding shape):", transformer_params['embedding'].shape)

    # --- Create Positional Encoding ---
    # This is created once and reused.
    positional_encoding = create_positional_encoding(config.max_seq_len, config.d_model)
    print(f"Positional Encoding shape: {positional_encoding.shape}")


    # --- Run Generation Example (with initial random parameters) ---
    _ = run_generation_example(config, transformer_params, positional_encoding, key_gen)
    
    # --- Run Training Loop Example ---
    # This will modify transformer_params
    trained_params = run_training_loop_example(config, transformer_params, positional_encoding, key_train)
    
    # --- Run Generation Example (with trained parameters) ---
    print("\n--- Running Generation Example with TRAINED parameters ---")
    key_gen_after_train, _ = random.split(key_gen) # Use a new key for generation
    _ = run_generation_example(config, trained_params, positional_encoding, key_gen_after_train)

    print("\nScript finished.")