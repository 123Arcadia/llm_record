import json
from pathlib import Path

import requests
import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
     load_model_and_tokenizer,
    render_prompt,
    extract_final_candidate,
    grade_answer
)
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache, top_p_filter
)
from reasoning_from_scratch.qwen3 import KVCache
from torch import nn


def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2) # 2 个空格进行格式化缩进
    return data

@torch.inference_mode ()
def sample_response(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.9
):
    input_ids = torch.tensor(tokenizer.encode(prompt), device= device)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits /= temperature
        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas.cpu(), 1).to(device)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        generated.append(next_token.item())
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype), ]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)



def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # Require \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, devcie="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids+answer_ids, devcie=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=1)
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1 # seq_len - 1
    t_ids = torch.arange(start, end ,devcie=devcie)
    next_tokens = full_ids[start + 1, end + 1]
    next_tokens_logps = logprobs[t_ids, next_tokens]
    return next_tokens_logps.mean().item()

def sequence_logprob_draft(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    # Positions whose next-token probabilities we want
    # These correspond to predicting token_ids[t + 1] from position t
    start = prompt_len - 1
    end = token_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=token_ids.device)
    next_tokens = token_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    # Sum log-probabilities over the answer tokens
    return torch.sum(next_token_logps)


def sequence_logprob(model, token_ids, prompt_len):
    # token_ids: prompt_ids + answer_ids
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)

    return torch.sum(selected[prompt_len-1:])


def compute_grpo_loss(model, tokenizer, examlple, device, num_rollouts=2, max_new_tokens=256, temperature=0.8, top_p=0.9):
    assert  num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []

    prompt = render_prompt(examlple["problem"])
    was_training = model.training
    model.train()

    for _ in range(num_rollouts):
        # Stage 1: generate rollouts
        token_ids, prompt_ids, text = sample_response(model ,tokenizer, prompt, device, max_new_tokens, temperature, top_p)
        # Stage 2: compute rewards
        reward = reward_rlvr(text, examlple["answer"])
        # Stage 3: compute log-probabilities
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append({
            "text": text,
            "reward": reward,
            "gen_len": token_ids.numel() - prompt_len
        })

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)
    # detach: 我们需要 detach()，因为我们希望将优势视为固定的训练信号;这样可以确保我们仅通过logprobs 进行反向传播。
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def save_checkpoint(model, checkpoint_dir, step, suffix):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
            checkpoint_dir /
            f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train_rlvr_grpo(model,
                    tokenizer,
                    math_data,
                    device,
                    steps=None,
                    num_rollouts=2,
                    max_new_tokens=256,
                    temperature=0.8,
                    top_p=0.9,
                    lr=1e-5,
                    checkpoint_every=50,
                    checkpoint_dir="."):

    if steps is None:
        steps = len(math_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    current_step = 0
    try:
        for step in range(steps):
            optimizer.zero_grad()
            current_step += 1
            example = math_data[step % len(math_data)]

            stats = compute_grpo_loss(model, tokenizer, example, device,
                                      num_rollouts, max_new_tokens, temperature, top_p)
            stats["loss_tensor"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            print(f"[step {current_step}/{steps}] Loss {stats["loss"]:.4f} reward_avg {reward_avg:.3f}")

            # 对生成 样本抽样检查
            if current_step % 10 == 0:
                print(f'[Step {current_step}] smaple outputs')
                for i,sample in enumerate(stats["samples"][:3]):
                    t = sample["text"].replace("\n", "\\n")
                    print(f"f{i+1} reward={sample['reward']:.3f} len={sample['gen_len']}: {t}")
                    print()

            if checkpoint_every and current_step % checkpoint_every == 0:
                ckp_path = save_checkpoint(model, checkpoint_dir, step=current_step)
                print(f'save checkpoint to {ckp_path}')


    except KeyboardInterrupt as e:
        ckp_path = save_checkpoint(model, checkpoint_dir, step=max(1, current_step), suffix="interrupt")
        print(f'KeyboardInterrupt: save checkpoint to {ckp_path}')
        return model
    return model



if __name__ == '__main__':
    print("====================加载数据========================")

    math_train = load_math_train(save_copy=True)
    print(f'Dataset size: {len(math_train)}')
    print(f'{math_train[0]=}')


    device = get_device()
    device = torch.device("cpu")

    model, tokenizer = load_model_and_tokenizer(
        which_model="base",
        device=device,
        use_compile=False
    )

    raw_prompt = (
        "Half the value of $3x-9$ is $x+37$. "
        "What is the value of $x$?"
    )
    prompt = render_prompt(raw_prompt)

    torch.manual_seed(0)
    response = generate_text_stream_concat_flex(
        model, tokenizer, prompt, device,
        max_new_tokens=2048, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.9,
        top_p=0.9
    )
    #  \boxed{58}  #实际答案：83
    print(f'=======generate_text_stream_concat_flex=========')
    print(f'{response=}')
    token_ids, prompt_len, answer_text = sample_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.9,
    )
    print(f'=======sample_response 1=========')
    print(f'{answer_text=}')


    print(f'=======sample_response 2=========')
    torch.manual_seed(5)

    token_ids, prompt_len, answer_text = sample_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=512,
                temperature=0.9,
                top_p=0.9,
            )

    print(f'{answer_text=}')

    print("==========对每个res进行打分reward=============")
    # 假设多次输出的res如下
    rollouts = [
        r"\boxed{83}",
        r"The correct answer is \boxed{83}",
        r"The final answer is 83",
        r"We get \boxed{38}",
    ]
    rollout_rewards = []
    for answer in rollouts:
        reward = reward_rlvr(answer_text=answer, ground_truth="83")
        print(f"Answer: {answer!r}")
        print(f"Reward: {reward}\n")
        rollout_rewards.append(reward)

    rewards = torch.tensor(rollout_rewards, device=device)
    print(f'{rewards=}')
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    print(f'{advantages=}')
    # tensor([ 0.8659,  0.8659, -0.8659, -0.8659])

    print("=============计算log_probs===============")
    avg_logprob_val = avg_logprob_answer(model,
                                         tokenizer,
                                        prompt=prompt,
                                        answer=answer_text,
                                        device=device)
    print(f'{avg_logprob_val=}')

    # 然而，GRPO 使用序列级对数概率，而非如上所述的长度归一化词元级平均值。·令牌级平均值对于评分很有用，因为它们使不同长度的输出可比较。在 GRPO 中，每个展开步骤都会在整个序列中获得一次奖励和一次优势。为了正确调整梯度的大小，对数概率必须反映出整个序列的概率，而这一概率是通过相加词级对数概率来获得的否则，平均对数概率会隐式地按序列长度重新缩放学习信号，并扭曲策略更新，尤其是在较长的rollout 中。
    # ·我们可以通过取消平均操作，并将torch.mean(next_token_logps)替换为torch.sum(nexttoken_logps)，将其转换为序列级对数概率。
    # 回顾性地，我们也可以将平均结果乘以答案令牌的数量来获得未平均的结果。.
    # notes: 其实就在函数中替换为sum操作即可
    sequence_logprob_val = avg_logprob_val * (len(tokenizer.encode(answer_text)))
    print(f'{sequence_logprob_val=}')

    print(f'{sequence_logprob_draft(model, token_ids, prompt_len)=}')

    print("=============计算log_probs 测试 ===============")
    rollouts = [
            r"\boxed{83}",
            r"The correct answer is \boxed{83}",
            r"The final answer is 83",
            r"We get \boxed{38}",
    ]
    print("=============计算log_probs(使用gather后) 测试 ===============")
    rollout_logps = []
    for text in rollouts:
        token_ids = tokenizer.encode(prompt + " " +text)
        logporb = sequence_logprob(model, torch.tensor(token_ids, device=device), prompt_len)
        print(f"Answer: {text}")
        print(f"Logporb: {logporb.item():.4f}\n")
        rollout_logps.append(logporb)

    logps = torch.stack(rollout_logps)
    print(f'{logps=}')


    print("=============计算 advantages 和 probs 的 loss ===============")
    pg_loss = -(advantages.detach() * logps).mean()
    print(f'{pg_loss=}')
    # tensor(-2.5764, grad_fn=<NegBackward0>)


    print("=============把loss计算总结成func ===============")
    stats = compute_grpo_loss(model=model, tokenizer=tokenizer, examlple=math_train[4], device=device, num_rollouts=2, max_new_tokens=256, temperature=0.8, top_p=0.9)
    print(f"{stats=}")
    torch.manual_seed(123)


    print("=============结合到train loop中 ===============")
    model = model.to(device)
    train_rlvr_grpo(model=model, tokenizer=tokenizer, math_data=math_train, device=device,
                    steps=50, num_rollouts=4, max_new_tokens=512, temperature=0.8, top_p=0.9,
                    lr=1e-5, checkpoint_every=5, checkpoint_dir=".")






















