"""
束宽度（Beam Width）
    定义：每步保留的候选序列数量
    权衡：
    宽度=1：等价于贪婪解码
    宽度越大：搜索空间越大，质量越高，但计算成本也越大
累积概率
    计算方式：序列概率 = 各个token概率的乘积
    数值稳定性：通常使用对数概率求和
    公式：log P(sequence) = Σ log P(token_i | context)

"""
import torch
import torch.nn.functional as F


def beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int,
                temperature: float = 1.0, top_k: int = None, stop_id: int = None) -> torch.Tensor:
    """
       束搜索：维护多个候选序列，选择最优路径

       Args:
           idx: 输入序列，形状为 (batch_size, seq_len)
           max_new_tokens: 最大生成token数量
           num_beams: 束宽度，表示保留的候选路径数量
           temperature: 温度参数，控制分布的平滑程度
           top_k: top-k过滤参数，限制候选token范围
           stop_id: 停止生成的token ID，遇到则停止

       Returns:
           生成的token序列，形状为 (batch_size, generated_length)
       """
    # 1. 初始化
    beams = [idx.clone() for _ in range(num_beams)]
    beam_scores = torch.zeros(num_beams, device=idx.device)
    beam_scores[0] = 0.0  # # 第一个候选是原始序列
    beam_scores[1:] = float('-inf')  # 其他候选初始分数为负无穷

    # 2. 主循环：逐步生成token
    for step in range(max_new_tokens):
        new_beams = []
        new_scores = []

        # 32. 扩展每个候选序列
        for beam_idx, beam in enumerate(beams):
            if beam_scores[beam_idx] == float('-inf'):
                continue  # 跳过无效候选

            # 前向传播获取logits
            output = self(beam)
            logits = output.logits[:, -1, :]

            # 应用温度和top-k
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('-inf')

            # 计算对数概率
            log_probs = F.log_softmax(logits, dim=1)
            # 获取前num_beams个候选token
            top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)

            # 4. 为当前候选生成多个扩展
            for k in range(num_beams):
                token = top_indices[:, k:k + 1]
                log_probs = top_log_probs[:, k]

                new_beam = torch.cat([beam, token], dim=1)
                new_score = beam_scores[beam_idx] + log_probs.item()

                new_beams.append(new_beam)
                new_scores.append(new_score)

        # 5. 筛选最佳候选
        if not new_beams:
            break

        # 按分数排序，选择前num_beams个
        sorted_indices = sorted(range(len(new_scores)), key=lambda i: new_scores[i], reverse=True)
        beams = [new_beams[i] for i in sorted_indices[:num_beams]]
        beam_scores = [new_scores[i] for i in sorted_indices[:num_beams]]

        # 检查停止条件
        if stop_id is not None and beams[0][0, -1] == stop_id:
            break
    # 6. 返回最佳序列
    return beams[0][:, idx.shape[1]:]  # 只返回生成部分
