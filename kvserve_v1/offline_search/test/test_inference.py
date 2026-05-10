import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Infer_Comm.src.cache.cache_utils import CustomCacheConfig, CustomCache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import pandas as pd
from datasets import load_dataset

BASE_MODEL_DIR = Path("/home/bingxing2/home/scx9kvs/mxy/models")
DATASET_NAME = "Xnhyacinth/LongBench"
DATASET_CONFIG = "2wikimqa"
DATASET_SPLIT = "test"

model_dir = Path(BASE_MODEL_DIR) / "Meta-Llama-3.1-8B-Instruct"
df = pd.read_csv("../duo_config/Meta-Llama-3.1-8B-Instruct_scores.csv", header=None).dropna()
scores = torch.tensor(df.values, dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    use_cache=True,
    # output_attentions=True,
    attn_implementation="flash_attention_2",
)

cache_config = CustomCacheConfig(
    scores=scores,
    heads_selection=0.5,
    high_key_max_value=64,
    high_value_max_value=64,
    low_key_max_value=32,
    low_value_max_value=32,
    axis_key=[2],
    axis_value=[1, 3],
    device="cuda:0",
)
past_key_values = CustomCache(cache_config=cache_config)

dataset = load_dataset(DATASET_NAME, data_dir=DATASET_CONFIG, split=DATASET_SPLIT)[0]
input_text = dataset.get('context', '') or dataset.get('text', '')
# inputs是一个字典，包括ids和mask
# 两者的shape是(bsz, seq_len)
input_text += "\n\nAnswer the question based on the context above.\n" + dataset.get('question', '')
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# 需要加入return_dict_in_generate参数才可以正常返回past_key_values和attentions, attentions是softmax后的结果
# 其中past_key_values的shape是(layers, k or v, bsz, kv_heads, seq_len, head_dim)，其中前两维是tuple，通过[]访问，后四维是tensor
# 其中attentions的shape是(new_seq_len, layers, bsz, heads, query_len, key_len), 其中前两维是tuple，通过[]访问，后四维是tensor
# new_seq_len是新生成token的数量
# query_len是query的长度(第一次迭代大小是input_seq_len，其余都是1)
# key_len是key的长度(第一次迭代大小是input_seq_len，之后每次迭代增一)
model.eval()
outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    past_key_values=past_key_values,
    return_dict_in_generate=True,
    # do_sample=True,
    # temperature=0.6,
    # top_k=40,
    # top_p=0.95,
)

answers = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("question: ", dataset.get('question', ''))
print("--------------------------------")
print("output answer: ", answers)
print("--------------------------------")
print("real answer: ", dataset.get('answers', ''))
print("--------------------------------")
