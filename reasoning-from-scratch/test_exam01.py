import collections
from pathlib import Path
from time import sleep
from typing import Counter

import torch
import math
import matplotlib.pyplot as plt
from reasoning_from_scratch.ch06 import reward_rlvr


def test_nonZero():
    eos_id = 11
    row_tokens = torch.tensor([
        [1,2,11,3],
        [5,11,9,4],
    ]) # (2, 4)
    bnonzero=  (row_tokens == eos_id).nonzero(as_tuple=True)
    print(f'{bnonzero=}')
    eos_pos = bnonzero[0]
    print(f'{eos_pos=}')
    # bnonzero=(tensor([0, 1]), tensor([2, 1]))
    # eos_pos=tensor([0, 1])

    # if len(eos_pos) > 0:
    #     row_tokens = row_tokens[:eos_pos[0]]


def test_eq_eos_ids():
    a = torch.tensor([1, 2, 3, 4, 11, 1, 43, 11])
    print(a == 11)
    # tensor([False, False, False, False,  True, False, False,  True])


def test_time():
    import time
    s = time.time()
    sleep(2)
    e = time.time()
    print(e - s)  # s


def test_a_unsqueeze():
    a = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(30)]
    print(torch.stack(a))


def test_lplot_log():
    a = torch.arange(-20, 20, dtype=torch.float)
    b = torch.log_softmax(a, dim=-1)
    plt.plot(a, b)
    plt.show()


def test_mean_std():
    a = torch.tensor([1, 1, 0, 0], dtype=torch.float)
    advat = (a - a.mean()) / (a.std() + 1e-4)
    print(advat)  # tensor([ 0.8659,  0.8659, -0.8659, -0.8659])


def test_rollouts_gead_anser():
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
        # Answer: '\\boxed{83}'
        # Reward: 1.0
        #
        # Answer: 'The correct answer is \\boxed{83}'
        # Reward: 1.0
        #
        # Answer: 'The final answer is 83'
        # Reward: 0.0
        #
        # Answer: 'We get \\boxed{38}'
        # Reward: 0.0


def test_ab():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 3])
    c = torch.tensor(a + b)
    print(c)
    # tensor([2, 4, 6])


def test_path():
    p = Path("../../data/math500_test.json")
    print(p.name)
    # math500_test.json


def test_brevity_plot():
    answer = torch.linspace(-1, 1, 100)
    brevity_bonus = 2
    score = -1.0
    scores = []
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    for i in range(len(answer)):
        score += 1.5 * math.exp(-len(answer[:i]) / brevity_bonus)
        scores.append(score)
    plt.plot(answer, scores, label="Brevity Penalty", marker="o")
    plt.xlabel("Answer Length")
    plt.ylabel("Score")
    plt.show()


def test_count_most():
    a = collections.Counter()
    a.update({"a1": 1})
    a.update({"b2": 2})
    a.update({"c3": 3})
    a["a1"] += 1

    most = a.most_common()
    print(F'{most=}')
    # most=[('c3', 3), ('a1', 2), ('b2', 2)]
