import mlx_parallm
import importlib
importlib.reload(mlx_parallm)

from mlx_parallm.utils import load, generate, batch_generate
import string
import random

# load model
model, tokenizer = load("google/gemma-1.1-2b-it")

capital_letters = string.ascii_uppercase
distinct_pairs = [(a, b) for i, a in enumerate(capital_letters) for b in capital_letters[i + 1:]]

num_prompts = 10
prompt_template = "Think of a real word containing both the letters {l1} and {l2}. Then, say 3 sentences which use the word."
prompts_raw = [prompt_template.format(l1=p[0], l2=p[1]) for p in random.sample(distinct_pairs, num_prompts)]
prompt_template_2 = "Come up with a real English word containing both the letters {l1} and {l2}. No acronyms. Then, give 3 complete sentences which use the word."
prompts_raw_2 = [prompt_template_2.format(l1=p[0], l2=p[1]) for p in random.sample(distinct_pairs, num_prompts)]

response = batch_generate(model, tokenizer, prompts=prompts_raw+prompts_raw_2, max_tokens=100, verbose=True, temp=0.0)