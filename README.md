# nanoGPT
 # TensorFlow GPT-2 Model

This repository contains the implementation of nanoGPT in TensorFlow. This is a conversion of the original PyTorch code from `nanoGPT` library by [Andery Karpathy](https://github.com/karpathy/nanoGPT). Special thanks to him and all contributors who made this possible.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Running the code](#running-the-code)
- [Docker Deployment](#docker-deployment)
- [GPT Config Explanation](#gpt-config-explanation)
- [Blog Post](#blog-post)

## Description

The code is designed to train a GPT model on text data. The text data is downloaded from a given URL, which by default points to a small portion of the Shakespeare dataset. The text is then encoded into a numerical format, which is used to train the model.

## Installation

To run this code, you need to have TensorFlow 2.4.0 or later installed on your system. You can install it via pip:

`pip install tensorflow`

## Running the code

You can run the code using the following command:
`python main.py`


## Docker Deployment

The code can be deployed using Docker. There are Dockerfile and docker-compose.yml files provided for this purpose. To deploy the Docker container, navigate to the directory containing the docker-compose.yml file and run:
`docker-compose up`



## GPT Config Explanation

The `GPTConfig` class contains the following fields:

- `block_size`: The size of the input block.
- `vocab_size`: The size of the vocabulary. For GPT-2, this is 50257.
- `n_layer`: The number of transformer blocks.
- `n_head`: The number of attention heads.
- `n_embd`: The size of the embeddings.
- `dropout`: The dropout rate for regularization.
- `bias`: If `True`, bias is included in linear and layer normalization operations. This is set to `True` to imitate GPT-2, but setting it to `False` can result in faster and better results.
- `epsilon`: The epsilon value for layer normalization.

## Blog Post

For more information about the code and how it works, check out the [LinkedIn post](https://www.linkedin.com/pulse/decoding-transformers-dive-gpt-tensorflow-krishna-chaitanya-kosaraju)




