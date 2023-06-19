import tensorflow as tf
from tensorflow.keras import layers
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to the nearest multiple of 64 for efficiency
    n_layer: int = 12 # number of squential transformers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding size of the input
    dropout: float = 0.2 # dropout percentage
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    epsilon: float = 1e-5  # epsilon value of layer normalization

class MultiHeadAttention(layers.Layer):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = config.n_head
        self.head_size = config.n_embd // config.n_head

        # Projecting input into key, query, and value for all attention heads, but in batch
        self.c_attn = layers.Dense(3 * config.n_embd, use_bias=config.bias)

        # Regularization
        self.attn_dropout = layers.Dropout(config.dropout)
        self.resid_dropout = layers.Dropout(config.dropout)
    
    def call(self, x):
        B, T, C = x.shape

        # Linear transformation for queries, keys, and values, note that C = n_embd
        qkv = self.c_attn(x)  # Input shape: (B, T, C), Output shape: (B, T, 3 * n_embd)

        # Split the queries, keys, and values
        q, k, v = tf.split(qkv, 3, axis=-1)  # Input shape: (B, T, 3 * n_embd), Output shapes: 3 * (B, T, n_embd)

        # Reshape queries, keys, and values for multi-head attention with head_size = n_embd // num_heads
        q = tf.reshape(q, (B, T, self.num_heads, -1))  # Output shape: (B, T, num_heads, head_size)
        k = tf.reshape(k, (B, T, self.num_heads, -1))  # Output shape: (B, T, num_heads, head_size)
        v = tf.reshape(v, (B, T, self.num_heads, -1))  # Output shape: (B, T, num_heads, head_size)

        # Perform attention operations

        # Transpose queries, keys, and values for efficient matrix multiplication
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # Output shape: (B, num_heads, T, head_size)
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # Output shape: (B, num_heads, T, head_size)
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # Output shape: (B, num_heads, T, head_size)

        # Compute attention scores ("affinities")
        wei = tf.matmul(q, k, transpose_b=True) * (self.head_size ** -0.5)  # Output shape: (B, num_heads, T, T)

        mask = tf.linalg.band_part(tf.ones_like(wei), -1, 0)  # Lower triangular matrix of ones
        wei = tf.where(mask == 1, wei, float("-inf"))  # Set upper triangular part to -inf

        wei = tf.nn.softmax(wei, axis=-1)  # Output shape: (B, num_heads, T, T)
        wei = self.attn_dropout(wei)  # Regularization step 1

        # Perform the weighted aggregation of the values
        out = tf.matmul(wei, v)  # Output shape: (B, num_heads, T, head_size)

        # Transpose and reshape the output to match the original shape
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # Output shape: (B, T, num_heads, head_size)
        out = tf.reshape(out, (B, T, -1))  # Output shape: (B, T, C) - note that C = num_heads * head_size = n_embd
        out = self.resid_dropout(out)  # Regularization step 2
        return out

class MLP(layers.Layer):
    def __init__(self, config):
        super(MLP, self).__init__()
        n_embed = config.n_embd
        self.c_fc = layers.Dense(4 * n_embed, use_bias=config.bias, activation=tf.keras.activations.gelu)
        self.c_proj = layers.Dense(config.n_embd, use_bias=config.bias)
        self.dropout = layers.Dropout(config.dropout)

    def call(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(layers.Layer):
    def __init__(self, config):
        super(Block, self).__init__()

        # Layer normalizing the input data as the number of features increases over time
        self.ln_1 = layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True)
        self.mlp = MLP(config)
    
    def call(self, x):
        # 1. Input data is layer normalized: Layer normalizing the input data as the number of features increases over time
        x_normalized = self.ln_1(x)
        
        # 2. Fed through the attention network: We get the attention scores or weighted values
        attn_output = self.attn(x_normalized)
        
        # 3. Added to the input: Reduces vanishing gradient issues
        x = x + attn_output
        
        # 4. Layer normalized the data again
        x_normalized = self.ln_2(x)
        
        # 5. Final pass through a multi-layer perceptron: We are learning the features
        mlp_output = self.mlp(x_normalized)
        
        # 6. Added to the input again
        x = x + mlp_output

        return x



class GPT(tf.keras.Model):
    def __init__(self, config):
        super(GPT, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = tf.keras.layers.LayerDict({
            # word token embeddings
            'wte': tf.keras.layers.Embedding(config.vocab_size, config.n_embd, input_length=config.block_size), 
            # word position embeddiings
            'wpe': tf.keras.layers.Embedding(config.block_size, config.n_embd),
            # dropoutt layer
            'drop': tf.keras.layers.Dropout(config.dropout),
            # Transformer blocks
            'h': tf.keras.Sequential([Block(config) for _ in range(config.n_layer)]),
            # layer normalization
            'ln_f': tf.keras.layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True)
        })

        # layer is used to project the output of the GPT model to the vocabulary size
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

        # sets the weights of the embedding layer (wte) in the transformer to be the transpose of the weights of the lm_head layer. 
        # This is done to initialize the embedding layer with the same weights as the lm_head layer, 
        # which helps in aligning the embeddings with the output projection. 
        # The transpose operation is necessary because the shapes of the weights in the lm_head layer and 
        # the embedding layer have different dimensions, and the transpose operation ensures compatibility between them.
        self.transformer['wte'].set_weights([self.lm_head.get_weights()[0].T])

        self._init_weights()

        num_params = self.get_num_params()
        print(f"number of parameters: {num_params / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(tf.reduce_prod(p.shape) for p in self.trainable_variables)
        if non_embedding:
            n_params -= tf.reduce_prod(self.transformer['wpe'].weights[0].shape)
        return n_params

    def _init_weights(self):
        for module in self.trainable_variables:
            if isinstance(module, tf.keras.layers.Dense):
                module.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
                if module.use_bias:
                    module.bias_initializer = tf.keras.initializers.Zeros()
            elif isinstance(module, tf.keras.layers.Embedding):
                module.embeddings_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def call(self, idx, targets=None):
        b, t = idx.shape # shape  is b, t
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = tf.range(0, t, dtype=tf.int64)  # shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer['wte'](idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer['wpe'](pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)
        for block in self.transformer['h'].layers:
            x = block(x)
        x = self.transformer['ln_f'](x)

        if targets is not None:
            # If we are given some desired targets, also calculate the loss

            # logit scores for each vocabulary word at each position in the input sequence.
            logits = self.lm_head(x) # shape (batch_size, sequence_length, vocab_size)

            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, -1, :]) # shape (b,)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # Model surgery to decrease the block size if necessary
        # E.g., we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # adjusts the weights of the 'wpe' embedding layer in the self.transformer model by cropping them to match the desired block_size.
        self.transformer['wpe'].set_weights([self.transformer['wpe'].get_weights()[0][:block_size]])

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden, see more notes below
        assert all(k == 'dropout' for k in override_args)

        # GPT2 model configurations
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        # Override default arguments
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints

        # Create a GPTConfig object
        config = GPTConfig(**config_args)

        # Create an instance of the GPT model
        model = cls(config)

        # Load the pretrained weights
        if model_type.startswith('gpt2'):
            checkpoint_path = 'path_to_pretrained_checkpoint'
            model.load_weights(checkpoint_path)

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay_params = []
        nodecay_params = []
        # Create optim groups. Any parameters that are 2D will be weight decayed, otherwise not.
        # i.e., all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        for var in self.model.trainable_variables:
            if var.shape.ndims >= 2:
                decay_params.append(var)
            else:
                nodecay_params.append(var)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(tf.reduce_prod(tf.shape(p)) for p in decay_params)
        num_nodecay_params = sum(tf.reduce_prod(tf.shape(p)) for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=betas[0], beta_2=betas[1])
        return optimizer



    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # First estimate the number of flops we do per iteration
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as a ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @tf.function
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in tf.range(max_new_tokens):
            # If the sequence context is growing too long, we must crop it at block_size
            if idx.shape[1] <= self.config.block_size:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                _, top_indices = tf.math.top_k(logits, k=min(top_k, logits.shape[-1]))
                mask = tf.broadcast_to(top_indices[:, -1:], logits.shape)
                logits = tf.where(logits < mask, float("-inf"), logits)

            # Apply softmax to convert logits to (normalized) probabilities
            probs = tf.nn.softmax(logits, axis=-1)

            # Sample from the distribution
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)

            # Append sampled index to the running sequence and continue
            idx = tf.concat((idx, idx_next), axis=1)

        return idx
