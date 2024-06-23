
# MLX ParaLLM

Batched KV caching for fast parallel inference on Apple Silicon devices, via [MLX](https://github.com/ml-explore/mlx). 

This repo heavily borrows from [`mlx_lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). Will explore how to add batched generation there as a non-breaking PR. 


## Usage
Requires `mlx` and `mlx_lm` to be installed.
```python
from mlx_parallm.utils import load, batch_generate
model, tokenizer = load("google/gemma-1.1-2b-it")
prompts = ["prompt_0", ..., "prompt_k"]
responses = batch_generate(model, tokenizer, prompts=prompts_raw[:10], max_tokens=100, verbose=True, format_prompts=True, temp=0.0)
```

## Models
Models tested: 
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-1.1-2b-it`
- `mlx-community/Phi-3-mini-4k-instruct-4bit`
- `mlx-community/gemma-1.1-2b-it-4bit`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit`

Both quantized and `float16` models are supported. `float16` models seem to generally perform faster if sufficient RAM is available (up to 1300+ tok/s throughput for `gemma-2b` on M3 Max 128GB).

Additional models can be added by copying architecture files from [`mlx_lm/models`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm/models) and replacing any references to `KVCache` with `BatchedKVCache`. 

## Features
Supported:
- `batch_generate` method (tested with `len(prompts) > 500`)
- Auto-padding
- Auto-formatting with prompt templates (`format_prompts=True`)
- `temp = 0`, `temp > 0`, `top_p` sampling
- single-stream `generate` method 

Not (yet) supported: 
- Repetition penalties
- Streaming outputs for `batch_generate`
- Dynamic batching for async requests