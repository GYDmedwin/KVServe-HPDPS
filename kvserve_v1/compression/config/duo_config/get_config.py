import os
import torch
import pandas as pd

from kvserve_v1.utils.logger import log_info


class DuoConfigGenerator:
    """
    Class to generate and retrieve DuoAttention configuration scores.
    """

    @staticmethod
    def duo_attention_on_the_fly(model, num_samples=None, q_len=1024, max_tokens=2048):
        """
        Compute DuoAttention scores on-the-fly from BookSum samples.
        Heavy dependencies (tqdm, datasets, transformers) are imported lazily.
        """
        from tqdm import tqdm
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        num_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        num_key_value_groups = num_heads // num_key_value_heads

        dataset = load_dataset("kmfoda/booksum", split="train").to_pandas()

        if num_samples and num_samples > 0:
            texts = dataset.sample(num_samples, random_state=42)["chapter"].tolist()
            num_texts = num_samples
        else:
            texts = dataset["chapter"].tolist()
            num_texts = len(texts)

        position_ids = torch.arange(q_len).unsqueeze(0)
        scores = torch.zeros(
            (model.config.num_hidden_layers, num_key_value_heads), dtype=torch.float32
        )

        for text in tqdm(texts, desc="Computing DuoAttention scores"):
            with torch.no_grad():
                inputs = tokenizer(
                    text, return_tensors="pt", max_length=max_tokens, truncation=True
                ).to(model.device)
                hidden_states = list(
                    model(**inputs, output_hidden_states=True).hidden_states[:-1]
                )

                for layer_idx, h in enumerate(hidden_states):
                    module = model.model.layers[layer_idx]
                    d = module.self_attn.head_dim
                    h = module.input_layernorm(h)

                    q = module.self_attn.q_proj(h)
                    q = q.view(1, q.shape[1], -1, d)
                    q = q.mean(dim=1, keepdim=True)
                    q = q.repeat(1, q_len, 1, 1).transpose(1, 2)

                    k = module.self_attn.k_proj(h)
                    k = k.view(1, k.shape[1], -1, d)
                    k = k.mean(dim=1, keepdim=True)
                    k = k.repeat(1, q_len, 1, 1).transpose(1, 2)

                    cos, sin = model.model.rotary_emb(h, position_ids.to(h.device))
                    q, k = apply_rotary_pos_emb(
                        q, k, cos.to(q.device), sin.to(q.device)
                    )
                    k = k.repeat_interleave(num_key_value_groups, dim=1)

                    attn_weights = torch.matmul(
                        q[:, :, -1:, :], k.transpose(2, 3)
                    ) / (d**0.5)
                    attn_weights = attn_weights.softmax(
                        dim=-1, dtype=torch.float32
                    ).squeeze()

                    s = torch.cumsum(attn_weights, dim=1, dtype=torch.float32).mean(1)
                    s = s.view(-1, num_key_value_groups).mean(1)
                    scores[layer_idx] += s.cpu() / num_texts

                del hidden_states
                torch.cuda.empty_cache()

        return scores.numpy()

    @classmethod
    def get_scores_from_csv(cls, model_name: str):
        """
        Retrieve scores from a CSV file.
        Search order: directory of this file → cwd → relative path.
        """
        model_basename = model_name.strip("/").split("/")[-1]
        csv_filename = f"{model_basename}_scores.csv"

        paths_to_check = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename),
            os.path.join(os.getcwd(), csv_filename),
            csv_filename,
        ]

        for path in paths_to_check:
            if os.path.exists(path):
                log_info(f"Loading DuoAttention scores from {path}")
                try:
                    df = pd.read_csv(path, header=None).dropna()
                    return torch.tensor(df.values, dtype=torch.float32)
                except Exception as e:
                    log_info(f"Error reading scores file {path}: {e}")
                    return None

        log_info(
            f"Scores file not found for {model_basename}. Checked: {paths_to_check}"
        )
        return None
