
## 评估model的一些方法

- letter逐个对比

- 使用ELo评分

  ```python
  def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):
    # Step 1: 初始化所有模型的ELO评分（默认1000）
    ratings = {
        model: initial_rating
        for pair in vote_pairs  # 遍历每一组胜负对
        for model in pair       # 提取胜负对中的两个模型
    }
    # Step 2: 遍历每一组胜负对，更新ELO评分
    for winner, loser in vote_pairs:
        # Step 3: 计算赢家的预期胜率（经典ELO公式）
        expected = 1.0 / (
            1.0 + 10 ** (
                (ratings[loser] - ratings[winner]) / 400.0
            )
        )
        # Step 4: 更新赢家的评分：k_factor * (实际得分 - 预期得分)
        # 实际得分=1（赢），所以 1 - expected
        ratings[winner] += k_factor * (1 - expected)
        # Step 5: 更新输家的评分：k_factor * (实际得分 - 预期得分)
        # 实际得分=0（输），输家的预期得分是 1 - expected，所以 0 - (1 - expected)
        ratings[loser] += k_factor * (0 - (1 - expected))
    return ratings
  ```
  

- 使用bradley_terry_leaderboard训练一个模型
    
    ```python
def bradley_terry_torch(vote_pairs, device):

    # Collect all unique model names
    models = sorted({m for winner, loser in vote_pairs for m in (winner, loser)})
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # Convert to index tensors
    winners = torch.tensor([idx[winner] for winner, _ in vote_pairs], dtype=torch.long)
    losers = torch.tensor([idx[loser] for _, loser in vote_pairs], dtype=torch.long)

    # Learnable parameters
    theta = torch.nn.Parameter(torch.zeros(n - 1, device=device))
    optimizer = torch.optim.Adam([theta], lr=0.01, weight_decay=1e-4)

    def scores():
        return torch.cat([theta, torch.zeros(1, device=device)])

    for epoch in range(500):
        s = scores()
        delta = s[winners] - s[losers]       # score difference
        loss = -torch.nn.functional.logsigmoid(delta).mean()   # negative log-likelihood
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Convert latent scores to Elo-like scale
    with torch.no_grad():
        s = scores()
        scale = 400.0 / math.log(10.0)
        R = s * scale
        R -= R.mean()
        R += 1000.0  # center around 1000

    return {m: float(r) for m, r in zip(models, R.cpu().tolist())}
```


- mmlu