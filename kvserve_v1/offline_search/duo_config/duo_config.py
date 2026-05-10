import os
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

MODEL_BASE_PATH = "/home/bingxing2/home/scx9kvs/mxy/models"

def duo_attention_on_the_fly(model, num_samples=None, q_len=1024, max_tokens=2048):
    """
    New experimental method to quickly compute DuoAttention scores:
    - Compute the mean query and key on num_samples random samples from BookSum. If num_samples is None or <= 0, all samples are used.
    - The input texts are truncated to max_tokens.
    - Repeat the mean query and key q_len times and apply RoPE to get (Q, K)
    - Compute the attention weights for (Q[-1], K) and compute the "area under the cumulated attention curve"
    These scores could also be saved to avoid recomputing them but this method is still experimental
    """

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    # Load data
    dataset = load_dataset("kmfoda/booksum", split="train").to_pandas()
    # dataset = load_dataset("csv", data_files="/home/bingxing2/home/scx9kvs/mxy/datasets/booksum/train.csv")["train"].to_pandas()
    
    if num_samples and num_samples > 0:
        texts = dataset.sample(num_samples, random_state=42)["chapter"].tolist()
        num_texts = num_samples
    else:
        texts = dataset["chapter"].tolist()
        num_texts = len(texts)

    # Initialize variables
    position_ids = torch.arange(q_len).unsqueeze(0)
    scores = torch.zeros((model.config.num_hidden_layers, num_key_value_heads), dtype=torch.float32)

    # Compute scores
    for text in tqdm(texts):
        with torch.no_grad():
            # Compute hidden states
            inputs = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(model.device)
            hidden_states = list(model(**inputs, output_hidden_states=True).hidden_states[:-1])

            for layer_idx, h in enumerate(hidden_states):
                module = model.model.layers[layer_idx]
                d = module.self_attn.head_dim
                h = module.input_layernorm(h)

                # Mean query
                q = module.self_attn.q_proj(h)
                q = q.view(1, q.shape[1], -1, d)
                # if isinstance(module, (Gemma3Attention, Qwen3Attention)):
                #     q = module.q_norm(q)
                q = q.mean(dim=1, keepdim=True)
                q = q.repeat(1, q_len, 1, 1).transpose(1, 2)

                # Mean key
                k = module.self_attn.k_proj(h)
                k = k.view(1, k.shape[1], -1, d)
                # if isinstance(module, (Gemma3Attention, Qwen3Attention)):
                #     k = module.k_norm(k)
                k = k.mean(dim=1, keepdim=True)
                k = k.repeat(1, q_len, 1, 1).transpose(1, 2)

                # Apply RoPE
                cos, sin = model.model.rotary_emb(h, position_ids.to(h.device))
                q, k = apply_rotary_pos_emb(q, k, cos.to(q.device), sin.to(q.device))
                k = k.repeat_interleave(num_key_value_groups, dim=1)

                # Compute attention weights for the last token
                attn_weights = torch.matmul(q[:, :, -1:, :], k.transpose(2, 3)) / (d**0.5)
                attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).squeeze()

                # Compute score: area under the cumulated attention curve
                s = torch.cumsum(attn_weights, dim=1, dtype=torch.float32).mean(1)
                s = s.view(-1, num_key_value_groups).mean(1)

                # Store the scores
                scores[layer_idx] += s.cpu() / num_texts

            del hidden_states
            torch.cuda.empty_cache()       

    return scores.numpy()

if __name__ == "__main__":

    model_name = "Qwen2.5-7B"
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(MODEL_BASE_PATH, model_name),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    scores = duo_attention_on_the_fly(model)
    print(scores)

    # 将 scores 保存到 CSV 文件
    output_filename = f"{model_name}_scores.csv"
    df = pd.DataFrame(scores)
    df.to_csv(output_filename, header=False, index=False)
    print(f"Scores successfully saved to {output_filename}")