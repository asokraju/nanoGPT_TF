import tensorflow as tf
from tensorflow.keras import layers
from dataclasses import dataclass

import os
import pandas as pd
import argparse
import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
import requests

from tensorflow import shape as tf_shape
from tensorflow import exp as tf_exp
from tensorflow import square as tf_square
from tensorflow import reduce_sum, reduce_mean
from tensorflow import GradientTape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean, Metric
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
# from tensorflow.keras import saving


@dataclass
class GPTConfig:
    block_size: int = 25
    vocab_size: int = 200  # GPT-2 vocab_size of 50257, padded up to the nearest multiple of 64 for efficiency
    n_layer: int = 12 # number of squential transformers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding size of the input
    dropout: float = 0.2 # dropout percentage
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    epsilon: float = 1e-5  # epsilon value of layer normalization

# Function to download the dataset
def text_extractor(url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    # Request to fetch the tiny shakespeare dataset
    response = requests.get(url)
    # Checking if we got a valid response
    if response.status_code == 200:
        # Opening a file and writing the content of the response
        with open('input.txt', 'w') as file:
            file.write(response.text)
    else:
        print(f"Failed to get file with status code: {response.status_code}")
    # Reading the downloaded file
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Function to encode the text into numbers
def text_encoder(text):
    # Listing and sorting the unique characters in the text
    chars = sorted(list(set(text)))
    # Getting the total number of unique characters
    vocab_size = len(chars)
    print("".join(chars))
    print(vocab_size)
    # Creating mappings from characters to their corresponding numerical representations
    stoi = {ch:i for i, ch in enumerate(chars)}
    # Creating mappings from numbers to their corresponding characters
    itos = {i:ch for i, ch in enumerate(chars)}
    # Function to encode a string into a list of numbers
    encode = lambda s: [stoi[ch] for ch in s]
    # Function to decode a list of numbers back into a string
    decode = lambda l: "".join([itos[i] for i in l])
    print(encode("hii I am Krishna"))
    print("decoded: ", decode(encode("hii I am Krishna")))
    # Encoding the entire text into numbers
    series = encode(text)
    n = int(0.8*len(series))
    return series

# Function to create a windowed dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # Creating a tensorflow dataset from the encoded series
    dataset = tf.data.Dataset.from_tensor_slices(series)
    # Creating a windowed dataset with each window of size window_size + 1 and shifting the window by 1 after each step
    dataset = dataset.window(size=window_size+1, shift = 1, drop_remainder=True)
    # Flattening the dataset
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    # Splitting each window into features (all elements except the last) and target (the last element)
    dataset = dataset.map(lambda x: (x[:-1], x[1:]))
    # Shuffling the dataset
    dataset = dataset.shuffle(shuffle_buffer)
    # Batching the dataset and prefetching 1 batch at a time to improve performance
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset






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
        # BUG: possible issue with tensorflow, you can use tf.reshape(q, (B, T, self.num_heads, -1)), for tensorflow B is unknown: it will give an error
        q = tf.reshape(q, (-1, T, self.num_heads, self.head_size))  # Output shape: (B, T, num_heads, head_size)
        k = tf.reshape(k, (-1, T, self.num_heads, self.head_size))  # Output shape: (B, T, num_heads, head_size)
        v = tf.reshape(v, (-1, T, self.num_heads, self.head_size))  # Output shape: (B, T, num_heads, head_size)


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
        out = tf.reshape(out, (-1, T, C))  # Output shape: (B, T, C) - note that C = num_heads * head_size = n_embd
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


def decoder(config):
    """
    Creates an decoder model based on the provided configuration.

    Args:
        config: An object specifying the configuration parameters.

    Returns:
        decoder: A Keras Model object representing the encoder model.
    """

    # create a dict with all the layers we need
    transformer_dict = {
        # input layer
        'input': tf.keras.Input(shape=(config.block_size,)),
        # word token embeddings
        'wte': tf.keras.layers.Embedding(config.vocab_size, config.n_embd, input_length=config.block_size),
        # word position embeddings
        'wpe': tf.keras.layers.Embedding(config.block_size, config.n_embd),
        # dropout layer
        'drop': tf.keras.layers.Dropout(config.dropout),
        # Transformer blocks
        'h': tf.keras.Sequential([Block(config) for _ in range(config.n_layer)]),
        # layer normalization
        'ln_f': tf.keras.layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True),
        # layer used to project the output of the GPT model to the vocabulary size
        'lm_head': tf.keras.layers.Dense(config.vocab_size, use_bias=False)
    }

    # input
    idx = transformer_dict['input']
    pos = tf.range(0, config.block_size, dtype=tf.int64)  # shape (t)

    # Forward the GPT model itself
    tok_emb = transformer_dict['wte'](idx)  # token embeddings of shape (b, t, n_embd)
    pos_emb = transformer_dict['wpe'](pos)  # position embeddings of shape (t, n_embd)
    x = transformer_dict['drop'](tok_emb + pos_emb)
    for block in transformer_dict['h'].layers:
        x = block(x)
    x = transformer_dict['ln_f'](x)

    # logit scores for each vocabulary word at each position in the input sequence.
    logits = transformer_dict['lm_head'](x)  # shape (batch_size, sequence_length, vocab_size)

    # Create encoder model
    model = tf.keras.Model(inputs=idx, outputs=logits, name='encoder')

    return model


if __name__ == '__main__':
    config = GPTConfig
    text = text_extractor()
    series = text_encoder(text)
    n = len(series)
    train_dataset = windowed_dataset(series[:n], config.block_size, batch_size=250, shuffle_buffer=10)
    test_dataset = windowed_dataset(series[n:], config.block_size, batch_size=250, shuffle_buffer=10)

    

    # Create the decoder model
    decoder_model = decoder(config)

    # Compile and train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    epochs = 10

    decoder_model.compile(optimizer=optimizer, loss=loss_fn)
    history = decoder_model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)