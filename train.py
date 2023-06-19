from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class MultiHeadAttention(layers.Layer):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = config.num_heads
        self.head_size = config.head_size

        # projecting  input into key, query and value for all attentian heads,but in batchw
        self.c_attn = layers.Dense(3 * config.head_size, use_bias=False)
        
        # regularization
        self.attn_dropout = layers.Dropout(config.dropout)
        self.resid_dropout = layers.Dropout(config.dropout)
    def call(self, x):
        B, T, C = x.shape

        # Linear transformation for queries, keys, and values
        qkv = self.c_attn(x)  # Input shape: (B, T, C), Output shape: (B, T, 3 * head_size)

        # Split the queries, keys, and values
        q, k, v = tf.split(qkv, 3, axis=-1)  # Input shape: (B, T, 3 * head_size), Output shapes: 3 * (B, T, head_size)

        # Reshape queries, keys, and values for multi-head attention with a  -  head_size = C//num_heads
        q = tf.reshape(q, (B, T, self.num_heads, -1))  # Output shape: (B, T, num_heads, head_size)
        k = tf.reshape(k, (B, T, self.num_heads, -1))  # Output shape: (B, T, num_heads, head_size)
        v = tf.reshape(v, (B, T, self.num_heads, -1))  # Output shape: (B, T, num_heads, head_size)

        # Perform attention operations

        # Transpose queries, keys, and values for efficient matrix multiplication
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # Output shape: (B, num_heads, T, head_size)
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # Output shape: (B, num_heads, T, head_size)
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # Output shape: (B, num_heads, T, head_size)

        # Compute attention scores ("affinities")
        wei = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * (self.head_size ** -0.5)  # Output shape: (B, num_heads, T, T)

        mask = tf.linalg.band_part(tf.ones_like(wei), -1, 0)  # Lower triangular matrix of ones
        wei = tf.where(mask == 1, wei, float("-inf"))  # Set upper triangular part to -inf

        wei = tf.nn.softmax(wei, axis=-1)  # Output shape: (B, num_heads, T, T)
        wei = self.attn_dropout(wei) # regularization step 1
        # Perform the weighted aggregation of the values
        out = tf.matmul(wei, v)  # Output shape: (B, num_heads, T, head_size)

        # Transpose and reshape the output to match the original shape
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # Output shape: (B, T, num_heads, head_size)
        out = tf.reshape(out, (B, T, -1))  # Output shape: (B, T, C) # note that C = num_heads*head_size
        out = self.resid_dropout(out) # regularization step 2
        return out

class MLP(layers.Layer):
    def __init__(self, config):
        super().__init__()
        n_embed = config.n_embed
        self.c_fc = layers.Dense(4* config.n_embed, use_bias=config.bias, 
                                 activation=tf.keras.activations.gelu)
        self.c_proj  = layers.Dense(config.n_embd, use_bias=config.bias)
        self.dropout = layers.Dropout(config.dropout)

    def call(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x