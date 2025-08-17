import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        
        # keep the top-k tokens with the highest probabilities before sampling the probabilities
        if top_k is not None and top_k > 0:
            masked_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, top_k:]
            logprobs[:, masked_idx] = -mx.inf

        # keep the top-p tokens with the highest cumulative probabilities before sampling the probabilities
        if top_p is not None and top_p > 0:
            idx = mx.argsort(-logprobs, axis=-1)
            sorted_probs = logprobs[:, idx]
            cumulative_probs = mx.cumsum(mx.exp(sorted_probs))
            mask = cumulative_probs < top_p
            mask[..., 0] = True
            logprobs[:, idx] = mx.where(mask, sorted_probs, -mx.inf)
        
        logprobs = logprobs / temp
        return mx.random.categorical(logprobs, axis=-1)
    return sample
