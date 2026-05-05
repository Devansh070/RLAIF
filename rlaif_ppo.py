import os
import math
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DynamicCache,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device = torch.device("cuda")
torch.set_default_dtype(torch.bfloat16)

batch_size = 4
gradient_accumulation_steps = 8
max_seq_len = 768

base_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
rm_model_name   = "OpenAssistant/reward-model-deberta-v3-large-v2"

gamma             = 1.0
gae_lambda        = 0.95
kl_coeff          = 0.05
clip_range_policy = 0.2
clip_range_value  = 0.2

num_epochs_sft = 1
num_epochs_ppo = 5

log_file_sft = "loss_sft.log"
log_file_ppo = "reward.log"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_and_format_dataset():
    all_data  = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
    train_raw = all_data["train"]

    def format_to_chatml(example):
        return {
            "chosen": (
                f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['chosen']}<|im_end|>"
            ),
            "rejected": (
                f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['rejected']}<|im_end|>"
            ),
        }

    original_columns = train_raw.column_names
    return train_raw.map(format_to_chatml, remove_columns=original_columns)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
def _get_cosine_schedule(
    current_step: int,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    linear_warmup: bool = False,
    min_value: float = 0.0,
) -> float:
    if current_step < num_warmup_steps:
        if linear_warmup:
            return min(1.0, (current_step + 1) / (num_warmup_steps + 1))
        return 1.0
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1.0 - min_value) * scale + min_value


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------
def generate_token_by_policy(chat_data, model, tokenizer, max_seq_len):
    """
    Greedy token generation with attention cache and left-padded input.

    Parameters
    ----------
    chat_data   : dict with 'input_ids' and 'attention_mask' (batch, seq_len)
    model       : nn.Module
    tokenizer   : PreTrainedTokenizer
    max_seq_len : int

    Returns
    -------
    cur_iids : torch.Tensor (batch, seq_len)
    cur_mask : torch.Tensor (batch, seq_len)
    """
    batch_size = chat_data["input_ids"].shape[0]
    cur_iids   = chat_data["input_ids"]
    cur_mask   = chat_data["attention_mask"]

    proceed_flag    = torch.ones(batch_size, dtype=bool).to(device)
    cache_position  = None
    past_key_values = DynamicCache()

    while torch.any(proceed_flag):
        cur_seq_len         = cur_iids.shape[1]
        token_indices       = torch.arange(cur_seq_len, dtype=int).to(device)
        last_nonpad_indices = (token_indices * cur_mask).argmax(-1)

        if cache_position is None:
            cache_position = torch.arange(cur_seq_len, dtype=int, device=device)
            logits = model(
                input_ids=cur_iids,
                attention_mask=cur_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            ).logits.detach()
            logits = logits[torch.arange(batch_size).to(device), last_nonpad_indices, :]
        else:
            logits = model(
                input_ids=cur_iids[:, -1:],
                attention_mask=cur_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            ).logits.detach()
            logits = logits.squeeze(1)

        probs        = F.softmax(logits, dim=-1)
        selected_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_token_indices = last_nonpad_indices + proceed_flag.int()
        if next_token_indices.max() > cur_seq_len - 1:
            cur_iids = F.pad(cur_iids, (0, 1, 0, 0), mode="constant", value=tokenizer.pad_token_id)
            cur_mask = F.pad(cur_mask, (0, 1, 0, 0), mode="constant", value=0)

        cur_iids[proceed_flag, next_token_indices[proceed_flag]] = selected_ids[proceed_flag]
        cur_mask[proceed_flag, next_token_indices[proceed_flag]] = 1

        cache_position = cache_position[-1:] + 1

        not_lim      = cur_mask.sum(dim=1) < max_seq_len
        is_eos       = torch.logical_and(selected_ids == tokenizer.eos_token_id, proceed_flag.bool())
        proceed_flag = torch.logical_and(proceed_flag, torch.logical_and(not_lim, ~is_eos))

    return cur_iids, cur_mask


# ---------------------------------------------------------------------------
# Reward scoring (RLAIF — DeBERTa reward model)
# ---------------------------------------------------------------------------
def get_reward_scores(gen_iids, smollm_tokenizer, rm_model, rm_tok, device):
    """
    Score a batch of generated sequences with the DeBERTa reward model.

    Parameters
    ----------
    gen_iids          : torch.Tensor (batch, seq_len)
    smollm_tokenizer  : PreTrainedTokenizer  (SmolLM2)
    rm_model          : AutoModelForSequenceClassification
    rm_tok            : PreTrainedTokenizer  (DeBERTa)
    device            : torch.device

    Returns
    -------
    scores : torch.Tensor (batch,)
    """
    texts = smollm_tokenizer.batch_decode(gen_iids, skip_special_tokens=False)

    assistant_marker = "<|im_start|>assistant\n"
    user_marker      = "<|im_start|>user\n"
    end_marker       = "<|im_end|>"

    questions, answers = [], []
    for text in texts:
        u_start = text.find(user_marker)
        u_end   = text.find(end_marker)
        question = text[u_start + len(user_marker): u_end].strip() if u_start != -1 and u_end != -1 else ""

        a_start = text.rfind(assistant_marker)
        if a_start != -1:
            answer_raw = text[a_start + len(assistant_marker):]
            a_end      = answer_raw.find(end_marker)
            answer     = answer_raw[:a_end].strip() if a_end != -1 else answer_raw.strip()
        else:
            answer = text.strip()

        questions.append(question)
        answers.append(answer)

    enc = rm_tok(
        questions,
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        scores = rm_model(**enc).logits.squeeze(-1)

    return scores


# ---------------------------------------------------------------------------
# Value model (critic)
# ---------------------------------------------------------------------------
class ValueModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.__setattr__(
            "lm_head",
            nn.Linear(576, 1, bias=False).to(device),
        )

    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits                   # (batch, seq_len, 1)
        return output.squeeze(-1)  # (batch, seq_len)


# ---------------------------------------------------------------------------
# GAE advantage
# ---------------------------------------------------------------------------
def get_advantage(delta, gamma, gae_lambda, seq_len):
    gae_params = torch.tensor(
        [(gamma * gae_lambda) ** i for i in range(seq_len)],
        dtype=torch.float32,
    ).to(device)
    adv = [
        torch.sum(delta[:, i:] * gae_params[: seq_len - i], dim=-1)
        for i in range(seq_len)
    ]
    return torch.stack(adv, dim=1)  # (batch, seq_len)


# ---------------------------------------------------------------------------
# Step 1 — Supervised Fine-Tuning
# ---------------------------------------------------------------------------
def run_sft(base_model, tokenizer, train_data):
    def collate_batch_sft(batch):
        itr_batch_size = len(batch)
        token_list   = [item["chosen"] for item in batch]
        token_tensor = tokenizer(
            token_list, padding=True, padding_side="right", return_tensors="pt"
        ).to(device)

        labels = token_tensor["input_ids"][:, 1:].clone()

        last_nonpad_indices = token_tensor["attention_mask"].sum(dim=1) - 1
        token_tensor["input_ids"][
            torch.arange(itr_batch_size).to(device), last_nonpad_indices
        ] = tokenizer.pad_token_id
        token_tensor["attention_mask"][
            torch.arange(itr_batch_size).to(device), last_nonpad_indices
        ] = 0
        inputs = token_tensor["input_ids"][:, :-1]
        masks  = token_tensor["attention_mask"][:, :-1]
        return inputs, labels, masks

    dataloader_sft = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch_sft
    )

    num_steps = math.ceil(len(dataloader_sft) / gradient_accumulation_steps)
    optimizer  = torch.optim.AdamW(base_model.parameters(), lr=9.0e-6, betas=(0.9, 0.999), eps=1e-08)
    scheduler  = LambdaLR(optimizer, lr_lambda=functools.partial(
        _get_cosine_schedule,
        num_training_steps=num_epochs_sft * num_steps,
        min_value=0.3,
    ))

    if os.path.exists(log_file_sft):
        os.remove(log_file_sft)

    for epoch in range(num_epochs_sft):
        base_model.train()
        optimizer.zero_grad()
        record_loss = []

        for i, (inputs, labels, masks) in enumerate(dataloader_sft):
            outputs = base_model(input_ids=inputs, attention_mask=masks)
            loss    = F.cross_entropy(outputs.logits.transpose(1, 2), labels)
            record_loss.append(loss.item())
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0 or i + 1 == len(dataloader_sft):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            print(
                f"Epoch {epoch+1} (iter{i+1}) "
                f"{math.ceil((i+1)/gradient_accumulation_steps)}/{num_steps} "
                f"- loss {loss:5.4f}",
                end="\r",
            )

        with open(log_file_sft, "a") as f:
            for l in record_loss:
                f.write(f"{l}\n")
        print()

    base_model.save_pretrained("./llm_sft")
    print("SFT done.")
    return dataloader_sft


def plot_sft_loss():
    with open(log_file_sft) as f:
        losses = [float(line) for line in f]
    interval = 50
    avg = [np.average(losses[i - interval + 1: i + 1]) for i in range(interval, len(losses))]
    plt.plot(np.arange(interval, len(losses)), avg)
    plt.title("SFT Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


def test_sft(tokenizer):
    messages = [
        "What do you most want to do right now?",
        "What is the best gift to give a friend who loves the outdoors?",
        "How do you relax after something bad happens?",
    ]
    inputs_test  = [f"<|im_start|>user\n{m}<|im_end|>\n<|im_start|>assistant\n" for m in messages]
    input_batch  = tokenizer(inputs_test, padding=True, padding_side="left", return_tensors="pt").to(device)
    input_seq_len = input_batch["input_ids"].shape[1]

    sft_model = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)
    sft_model.eval()

    with torch.no_grad():
        iids, _ = generate_token_by_policy(input_batch, sft_model, tokenizer, max_seq_len)

    iids    = iids[:, input_seq_len:]
    outputs = tokenizer.batch_decode(iids, skip_special_tokens=True)
    for i, msg in enumerate(messages):
        print("***** Question *****")
        print(msg)
        print("***** Answer (SFT) *****")
        print(outputs[i])
        print()

    del sft_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 2 — Load pre-trained reward model (RLAIF)
# ---------------------------------------------------------------------------
def load_reward_model():
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(rm_model_name).to(device)
    rm.eval()
    params_m = sum(p.numel() for p in rm.parameters()) / 1e6
    print(f"Reward model loaded — {params_m:.1f} M parameters")
    return rm, rm_tokenizer


# ---------------------------------------------------------------------------
# Step 3 — PPO training
# ---------------------------------------------------------------------------
def filter_by_length(train_data, tokenizer):
    def add_seq_len(example):
        def tok_len(text):
            return len(tokenizer(text)["input_ids"])
        return {"chosen_len": tok_len(example["chosen"]), "rejected_len": tok_len(example["rejected"])}

    train_data = train_data.map(add_seq_len)
    train_data = train_data.filter(
        lambda ex: ex["chosen_len"] <= max_seq_len and ex["rejected_len"] <= max_seq_len
    )
    return train_data.remove_columns(["chosen_len", "rejected_len"])


def run_ppo(policy_model, value_model, tokenizer, rm, rm_tokenizer, train_data):
    def rm_fin_msg(chat_str):
        target    = "<|im_start|>assistant\n"
        start_idx = chat_str.rfind(target)
        return chat_str[: start_idx + len(target)]

    def collate_batch_ppo(batch):
        chat_list = [rm_fin_msg(item["chosen"]) for item in batch]
        return tokenizer(chat_list, padding=True, padding_side="left", return_tensors="pt").to(device)

    dataloader_ppo = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch_ppo
    )
    num_steps = math.ceil(len(dataloader_ppo) / gradient_accumulation_steps)

    opt1 = torch.optim.AdamW(value_model.parameters(),  lr=3.0e-5, betas=(0.9, 0.999), eps=1e-08)
    sch1 = LambdaLR(opt1, lr_lambda=functools.partial(
        _get_cosine_schedule,
        num_training_steps=num_epochs_ppo * num_steps,
        num_warmup_steps=math.ceil(num_epochs_ppo * num_steps * 0.1),
    ))

    opt2 = torch.optim.AdamW(policy_model.parameters(), lr=3.0e-5, betas=(0.9, 0.999), eps=1e-08)
    sch2 = LambdaLR(opt2, lr_lambda=functools.partial(
        _get_cosine_schedule,
        num_training_steps=num_epochs_ppo * num_steps,
        num_warmup_steps=math.ceil(num_epochs_ppo * num_steps * 0.1),
        linear_warmup=True,
    ))

    if os.path.exists(log_file_ppo):
        os.remove(log_file_ppo)

    rm.eval()

    for epoch in range(num_epochs_ppo):
        opt1.zero_grad()
        opt2.zero_grad()
        record_reward = []

        for i, chat in enumerate(dataloader_ppo):
            itr_batch_size = chat["input_ids"].shape[0]
            input_seq_len  = chat["input_ids"].shape[1]

            # Phase A — rollouts (no gradient)
            policy_model.eval()
            with torch.no_grad():
                gen_iids, gen_mask = generate_token_by_policy(chat, policy_model, tokenizer, max_seq_len)

                seq_len       = gen_iids.shape[1]
                token_indices = torch.arange(seq_len, dtype=int).to(device)
                inf_mask      = (gen_mask * (token_indices >= input_seq_len).int()).bool()

                last_nonpad_indices = (token_indices * gen_mask).argmax(-1)
                for b in range(itr_batch_size):
                    shift = (seq_len - last_nonpad_indices[b] - 1).item()
                    gen_iids[b] = torch.roll(gen_iids[b], shifts=shift)
                    gen_mask[b] = torch.roll(gen_mask[b], shifts=shift)
                    inf_mask[b] = torch.roll(inf_mask[b], shifts=shift)

                first_nonpad = (torch.flip(token_indices, dims=(0,)) * gen_mask).argmax(-1)
                start_index  = first_nonpad.min()
                gen_iids = gen_iids[:, start_index:]
                gen_mask = gen_mask[:, start_index:]
                inf_mask = inf_mask[:, start_index:]
                inf_mask = inf_mask[:, :-1]

                rewards     = torch.zeros_like(gen_iids[:, :-1], dtype=torch.bfloat16).to(device)
                seq_rewards = get_reward_scores(gen_iids, tokenizer, rm, rm_tokenizer, device)
                rewards[:, -1] = seq_rewards
                record_reward.append(seq_rewards.mean().item())

                is_eos     = gen_iids[:, -1] == tokenizer.eos_token_id
                is_eos_num = is_eos.int().sum()
                if is_eos_num == 0:
                    continue
                if is_eos_num != itr_batch_size:
                    gen_iids = gen_iids[is_eos]
                    gen_mask = gen_mask[is_eos]
                    inf_mask = inf_mask[is_eos]
                    rewards  = rewards[is_eos]

            # Phase B — value model update
            value_model.train()
            with torch.set_grad_enabled(True):
                values_new  = value_model(input_ids=gen_iids[:, :-1], attention_mask=gen_mask[:, :-1])
                values_new  = values_new * gen_mask[:, :-1].float()
                values_old  = values_new.detach()
                values_next = F.pad(values_old[:, 1:], (0, 1, 0, 0), mode="constant", value=0.0)

                delta     = rewards + values_next * gamma - values_old
                adv_val   = get_advantage(delta, gamma, 1.0, delta.shape[1])
                values_actual = adv_val + values_old

                values_var    = torch.masked_select(torch.square(values_old - values_actual), inf_mask).mean()
                values_stddev = torch.sqrt(values_var)

                values_new_clipped = torch.clamp(
                    values_new,
                    values_old - clip_range_value * values_stddev,
                    values_old + clip_range_value * values_stddev,
                )
                val_loss = 0.5 * torch.max(
                    torch.square(values_new - values_actual),
                    torch.square(values_new_clipped - values_actual),
                ) / values_var
                val_loss = torch.masked_select(val_loss, inf_mask).mean()
                val_loss.backward()

                if (i + 1) % gradient_accumulation_steps == 0 or i + 1 == len(dataloader_ppo):
                    opt1.step()
                    sch1.step()
                    opt1.zero_grad()

            # Phase C — policy model update
            policy_model.train()
            with torch.set_grad_enabled(True):
                logits_new = policy_model(
                    input_ids=gen_iids[:, :-1], attention_mask=gen_mask[:, :-1]
                ).logits
                logits_old = logits_new.detach()

                logprb_old = -F.cross_entropy(logits_old.transpose(1, 2), gen_iids[:, 1:], reduction="none")
                logprb_new = -F.cross_entropy(logits_new.transpose(1, 2), gen_iids[:, 1:], reduction="none")
                prb_ratio  = torch.exp(logprb_new - logprb_old)
                prb_ratio_clipped = torch.clamp(prb_ratio, 1.0 - clip_range_policy, 1.0 + clip_range_policy)

                adv_pol  = get_advantage(delta, gamma, gae_lambda, delta.shape[1])
                pg_loss  = torch.masked_select(
                    torch.max(-adv_pol * prb_ratio, -adv_pol * prb_ratio_clipped),
                    inf_mask,
                ).mean()

                l_old     = logits_old - torch.amax(logits_old, dim=2, keepdim=True)
                l_new     = logits_new - torch.amax(logits_new, dim=2, keepdim=True)
                e_old     = torch.exp(l_old)
                e_new     = torch.exp(l_new)
                e_sum_old = torch.sum(e_old, dim=2, keepdim=True)
                e_sum_new = torch.sum(e_new, dim=2, keepdim=True)
                p_old     = e_old / e_sum_old
                kl_loss   = torch.masked_select(
                    torch.sum(p_old * (l_old - l_new + torch.log(e_sum_new) - torch.log(e_sum_old)), dim=2),
                    inf_mask,
                ).mean()

                total_loss = pg_loss + kl_loss * kl_coeff
                total_loss.backward()

                if (i + 1) % gradient_accumulation_steps == 0 or i + 1 == len(dataloader_ppo):
                    opt2.step()
                    sch2.step()
                    opt2.zero_grad()

            print(
                f"Epoch {epoch+1} (iter{i+1}) "
                f"{math.ceil((i+1)/gradient_accumulation_steps)}/{num_steps} "
                f"- reward {seq_rewards.mean().item():5.4f}",
                end="\r",
            )

        epoch_avg = sum(record_reward) / len(record_reward)
        print(
            f"Epoch {epoch+1} (iter{i+1}) "
            f"{math.ceil((i+1)/gradient_accumulation_steps)}/{num_steps} "
            f"- reward {epoch_avg:5.4f}"
        )
        with open(log_file_ppo, "a") as f:
            for r in record_reward:
                f.write(f"{r}\n")

    torch.save(value_model.state_dict(), "value.pt")
    policy_model.save_pretrained("./llm_aligned")
    print("PPO done.")


def plot_ppo_reward():
    with open(log_file_ppo) as f:
        reward_data = [float(line) for line in f]
    interval = 120
    avg = [np.average(reward_data[i - interval + 1: i + 1]) for i in range(interval, len(reward_data))]
    plt.plot(np.arange(interval, len(reward_data)), avg)
    plt.title("PPO Reward (moving average)")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.show()


def test_aligned(policy_model, tokenizer):
    messages = [
        "What do you most want to do right now?",
        "What is the best gift to give a friend who loves the outdoors?",
        "How do you relax after something bad happens?",
    ]
    inputs = [f"<|im_start|>user\n{m}<|im_end|>\n<|im_start|>assistant\n" for m in messages]
    input_batch   = tokenizer(inputs, padding=True, padding_side="left", return_tensors="pt").to(device)
    input_seq_len = input_batch["input_ids"].shape[1]

    policy_model.eval()
    with torch.no_grad():
        iids, _ = generate_token_by_policy(input_batch, policy_model, tokenizer, max_seq_len)

    iids    = iids[:, input_seq_len:]
    outputs = tokenizer.batch_decode(iids, skip_special_tokens=True)
    for i, msg in enumerate(messages):
        print("***** Question *****")
        print(msg)
        print("***** Answer (RLAIF-aligned) *****")
        print(outputs[i])
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_data = load_and_format_dataset()

    config     = AutoConfig.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, config=config).to(device)
    tokenizer  = AutoTokenizer.from_pretrained(base_model_name)

    run_sft(base_model, tokenizer, train_data)
    plot_sft_loss()
    test_sft(tokenizer)

    rm, rm_tokenizer = load_reward_model()

    train_data   = filter_by_length(train_data, tokenizer)
    policy_model = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)

    base_for_val = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)
    value_model  = ValueModel(base_for_val).to(device)

    run_ppo(policy_model, value_model, tokenizer, rm, rm_tokenizer, train_data)
    plot_ppo_reward()
    test_aligned(policy_model, tokenizer)


if __name__ == "__main__":
    main()
