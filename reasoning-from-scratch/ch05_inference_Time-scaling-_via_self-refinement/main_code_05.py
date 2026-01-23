import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache
)
from reasoning_from_scratch.ch03 import extract_final_candidate
import math


def heuristic_score(
        answer,
        prompt=None,  # Placeholder that is ignored
        brevity_bonus=500.0,
        boxed_bonus=2.0,
        extract_bonus=1.0,
        fulltext_bonus=0.0,
):
    score = 0.0

    # Reward answers that have a final boxed value
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus

    # Give weaker rewards if answer doesn't have a boxed value
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(
                answer, fallback="number_then_full"
            )
            if cand:
                score += fulltext_bonus

    # Add a brevity reward that decays with text length
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score



@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    # Get logits and probabilities similar to text generation functions
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    all_probas = torch.softmax(logits, dim=-1)

    # Positions we score (here: all)
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)

    # Since we have the text, we know the true next tokens
    next_ids = token_ids[1:]

    # Get probabilities for each next token
    next_token_probas = all_probas[t_idx, next_ids] # allprobas[[0, 1, 2], [1, 2, 4]]类似取all_probas[0, 1]、[1, 2]、[2, 4]的是哪个数据

    print(
        "Next-token probabilities:",
        [p.item() for p in next_token_probas]
    )

    # Likelihood of the sequence is the product of the probability scores
    print(
        "Joint probability:",
        torch.prod(next_token_probas)
    )

@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device):

    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    # We now use log_softmax
    all_logprobas = torch.log_softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]

    print(
        "Next-token log-probabilities:",
        [p.item() for p in next_token_logprobas]
    )
    # We replace the product with a sum
    print(
        "Joint log-probability:",
        torch.sum(next_token_logprobas)
    )


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    """
    对模型response进行打分
    """

    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device = device)
    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt) - 1 # answer第一个预测的位置
    end = prompt_ids.shape[0] - 1 # ids:[sel_len, vocab_size], 取seq_len-1位置

    # 需要得到strat, end的logprobs内容
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1, end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.mean(next_token_logps).item()

    score_1 = avg_logprob_answer(
        model, tokenizer,
        prompt="What is the capital of Germany?",
        answer=" The capital of Germany is Berlin.",
        device=device
    )
    print(f'{score_1=}')

    score_2 = avg_logprob_answer(
        model, tokenizer,
        prompt="What is the capital of Germany?",
        answer=" The capital of Germany is Bridge.",
        device=device
    )
    print(f'{score_2=}')

    prompt_cot = prompt + "\n\nExplain step by step."
    print(avg_logprob_answer(
                        model, tokenizer,
                        prompt=prompt_cot,
                        answer=response_1,
                        device=device
    ))
    print(avg_logprob_answer(
                        model, tokenizer,
                        prompt=prompt_cot,
                        answer=response_2,
                        device=device
    ))



if __name__ == '__main__':

    device = get_device()

    model, tokenizer = load_model_and_tokenizer(which_model="base", device=device, use_compile=False)

    raw_prompt = (
        "Half the value of $3x-9$ is $x+37$. "
        "What is the value of $x$?"
    )
    prompt = render_prompt(raw_prompt)
    prompt_cot = prompt + "\n\nExplain step by step."

    torch.manual_seed(0)
    response_1 = generate_text_stream_concat_flex(
        model, tokenizer, prompt_cot, device,
        max_new_tokens=2048, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.9,
        top_p=0.9
    )

    print(f'{response_1=}')

    torch.manual_seed(3)
    response_2 = generate_text_stream_concat_flex(
        model, tokenizer, prompt_cot, device,
        max_new_tokens=2048, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.9,
        top_p=0.9,
    )
    print(f'{response_2=}')

    print("Response 1 characters:", len(response_1))
    print("Response 1 tokens:", len(tokenizer.encode(response_1)))
    print("\nResponse 2 characters:", len(response_2))
    print("Response 2 tokens:", len(tokenizer.encode(response_2)))
    # Response 1 characters: 1422
    # Response 1 tokens: 537
    #
    # Response 2 characters: 651
    # Response 2 tokens: 284


    print(f'===========calc_next_token_probas================')
    print(calc_next_token_probas(
        model, tokenizer, device=device,
        prompt="The capital of Germany is Hamburg"
    ))



    # 5.6 Scoring model confidence with log-probabilities
    example_prompt = "What is the capital of Germany?"
    example_answer = " The capital of Germany is Berlin."

    calc_next_token_logprobas(
        model, tokenizer, device=device,
        prompt=example_prompt + example_answer
    )
    print(f'{len(tokenizer.encode(example_answer))=}')
