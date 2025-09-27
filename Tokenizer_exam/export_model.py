import torch
from transformers import AutoTokenizer

from llama2_exam.llama_attn_exam import ModelConfig, Transformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_model(tokenizer_path, model_config, model_ckp_path, save_directory):
    ModelConfig.register_for_auto_class()
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    # 初始化模型
    model = Transformer(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #加载权重
    state_dict = torch.load(model_ckp_path, map_location=device)
    unwanted_prefix = '_orig_mod'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    print(f'模型参数: {count_parameters(model)/1e6:.2f}M = {count_parameters(model)/1e9:.2f}B')

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    model.save_pretrained(save_directory, safe_serialization=False)
    tokenizer.save_pretrained(save_directory)
    print(f'模型、tokenizer以保存至: {save_directory}')



if __name__ == '__main__':
    config = ModelConfig(
        dim=1024, n_layers=18
    )

    export_model(
        tokenizer_path = './tokenizer_k',
        model_config = config,
        model_ckp_path = './BeelGroup_sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth',
        save_directory = 'k-model-215M'
    )