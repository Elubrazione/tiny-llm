import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    # You only need the last token's logits to decide the next token. 
    # Therefore, you need to select the last token's logits from the output logits.
    def _step(model, y, offset):
        out = model(y[None], offset)[:, -1, :]   # (N, L, E), here batch_size=1 by using y[None]
        output_probs = out - mx.logsumexp(out, keepdims=True)
        if sampler is None:
            return mx.argmax(output_probs, axis=-1)
        else:
            return sampler(output_probs)
    
    # prompt => tokens
    token_ids = mx.array(tokenizer._tokenizer.encode(prompt))   # shape (L,)
    detokenizer = tokenizer._detokenizer
    detokenizer.reset()

    while True:
        tk = _step(model, token_ids, 0) # Whatever the value of offset is
        mx.eval(tk)
        token_ids = mx.concat([token_ids, tk])
        if tk.item() == tokenizer._tokenizer.eos_token_id:
            break
        detokenizer.add_token(tk.item())
        print(detokenizer.last_segment, end="", flush=True)
        

def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    pass


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass
