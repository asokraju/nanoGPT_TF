from tensorflow.keras import layers
import tensorflow as tf
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to the nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding size of the input
    dropout: float = 0.0
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
        # 2. Fed through the attention network: We get the attention scores or weighted values
        # 3. Added to the input: Reduces vanishing gradient issues
        attn_output = self.attn(self.ln_1(x))
        x = x + attn_output
        
        # 4. Layer normalized the data again
        # 5. Final pass through a multi-layer perceptron: We are learning the features
        # 6. Added to the input again
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        return x
