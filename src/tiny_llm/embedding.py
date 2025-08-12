import mlx.core as mx
from .basics import linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight    # (vocab_size x embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        '''
        Input: N.. (tokens)
        Output: N.. x embedding_dim (vectors)
        '''
        # This can be done with a simple array index lookup operation.
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        '''
        In the Qwen2 model, the embedding layer can also be used as a linear layer to map the embeddings back to the token space.
        
        Input: N.. x embedding_dim
        Output: N.. x vocab_size
        '''
        return linear(x, self.weight)
