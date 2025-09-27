from contextlib import nullcontext


import torch

from transformers import AutoTokenizer

from llama2_exam.llama_attn_exam import Transformer, ModelConfig


class TextGenerator:
    def __init__(self,
                 checkpoint = './base_model_215M/pretrain_1024_18_6144.pth',
                 tokenizer_model_path='./tokenizer_k', # 分词模型路径
                 seed = 42,
                 device =None,
                 dtype = 'bfloat16' # # 数据类型，默认为 float32，可以选择 float16 或 bfloat16
                  ):
        """

        :param checkpoint:
        :param tokenizer_model_path:
        :param seed:
        :param device:
        :param dtype:  对torch.amp.autocast  'bf16' for cuda ; 'f16' for cpu
        """
        """
                初始化 TextGenerator 类，加载模型、设置设备和分词器等。
                """
        # 模型加载配置
        self.checkpoint = checkpoint
        self.tokenizer_model_path = tokenizer_model_path
        self.seed = seed
        self.device =  device or ('cuda:0' if torch.cuda.is_available() else 'cpu')  # 根据硬件条件选择设备
        self.dtype = dtype
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # 判断当前设备是否为 CUDA

        # 设置随机种子，确保生成的可重复性
        torch.manual_seed(seed)  # 设置 CPU 随机种子
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许 CUDA 使用 TF32 精度进行矩阵乘法运算
        torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32 精度加速

        # 根据 dtype 选择适当的自动混合精度上下文
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        # 加载ckp文件
        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)
        self.model = Transformer(ModelConfig(dim=1024, n_layers=18))
        sunwanted_prefix = '_orig_mod'
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix):]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model has {num_params / 1e6:.3f} M parameters.')
        # 设置模型为评估模式（evaluation mode），防止训练模式下的 dropout 等操作影响结果
        self.model.eval()
        self.model.to(self.device)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_path)

    def pretrain_sample(self,
                        start: str="Hello!", # 生成文本的起始提示词，可以是任意字符串
                        num_samples=3, # 生成样本的数量，默认生成 3 个样本
                        max_new_tokens=256, # 每个样本生成的最大 token 数，默认最多生成 256 个 token
                        temperature=0.7,  # 控制生成的随机性，1.0 为标准，值越大越随机
                        top_k = 300): # 保留概率最高的 top_k 个 token，限制生成时的选择范围
        """
                根据给定的起始文本生成样本。

                :param start: 生成文本的起始提示词
                :param num_samples: 要生成的文本样本数
                :param max_new_tokens: 每个样本生成的最大 token 数
                :param temperature: 控制生成的随机性，值越小生成越确定，值越大生成越随机
                :param top_k: 限制生成时选择的 token 范围
                :return: 生成的文本样本列表
        """
        # 如果 start 是以 'FILE:' 开头，表示从文件中读取起始文本
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read() # 读取文件内容作为起始文本

        # 起始文本编号是token id序列
        start_ids = self.tokenizer(start).data['input_ids']
        print(f'({len(start_ids)}) {start_ids=}')
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        print(f'input_ids 转为 tensor 的{x.shape=}')
        generated_texts = []# 保存生成的文本样本
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                    generated_texts.append(self.tokenizer.decode(y[0].tolist())) # 解码生成的token序列为可读文本

        return generated_texts



if __name__ == '__main__':
    print("------------------- Pretrain Sample ------------------- \n")

    pretrain_prompt_datas = [
        '<|im_start|>北京大学是',
        '<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院',
    ]
    generator = TextGenerator(checkpoint='./base_model_215M/pretrain_1024_18_6144.pth')
    for i in range(len(pretrain_prompt_datas)):
        samples= generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f"\nSample {i + 1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-' * 20}")  # 打印生成的样本并用分隔线分割

    print("\n ------------------- SFT Sample ------------------- \n")

    sft_prompt_datas = [
        '你好呀',
        "中国的首都是哪里？",
        "1+12等于多少？",
        "你是谁？"
    ]
    generator = TextGenerator(checkpoint='./sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth')  # 初始化生成器
    for i in range(len(sft_prompt_datas)):
        samples = generator.sft_sample(start=sft_prompt_datas[i], num_samples=1, max_new_tokens=128, temperature=0.6)
        print(
            f"\nSample {i + 1}:\nQuestion: {sft_prompt_datas[i]} \nAI answer: {samples[0]}\n{'-' * 20}")  # 打印生成的样本并用分隔线分割