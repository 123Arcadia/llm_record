import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

def test_decoding_strategies():
    """
    测试三种解码策略：贪婪解码、随机采样、束搜索
    """
    model_id = "../model/kmno4zx/happy-llm-215M-sft/"

    print("正在加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cpu").eval()

    # 测试prompt
    test_prompt = "请介绍一下自己"
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": test_prompt}
    ]

    # 准备输入
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_ids).data['input_ids']
    x = (torch.tensor(input_ids, dtype=torch.long)[None, ...]).to(model.device)

    print(f"测试prompt: {test_prompt}")
    print(f"输入token数量: {len(input_ids)}")
    print("=" * 60)

    # 测试1: 贪婪解码 (Greedy Search)
    print("🔍 测试1: 贪婪解码 (Greedy Search)")
    print("参数: do_sample=False, num_beams=1, temperature=0.0")
    print("特点: 每步选择概率最大的token，结果确定，速度快")

    with torch.no_grad():
        greedy_output = model.generate_super(
            x,
            stop_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            num_beams=1
        )
        greedy_response = tokenizer.decode(greedy_output[0].tolist(), skip_special_tokens=True)

    print(f"贪婪解码结果: {greedy_response}")
    print()

    # 测试2: 随机采样 (Random Sampling)
    print("🎲 测试2: 随机采样 (Random Sampling)")
    print("参数: do_sample=True, num_beams=1, temperature=0.8, top_k=50")
    print("特点: 基于概率分布随机采样，结果多样，创造性高")

    with torch.no_grad():
        # 运行多次以展示随机性
        for i in range(3):
            sampling_output = model.generate_super(
                x,
                stop_id=tokenizer.eos_token_id,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                do_sample=True,
                num_beams=1
            )
            sampling_response = tokenizer.decode(sampling_output[0].tolist(), skip_special_tokens=True)
            print(f"随机采样结果 {i+1}: {sampling_response}")

    print()

    # 测试3: 束搜索 (Beam Search)
    print("🔦 测试3: 束搜索 (Beam Search)")
    print("参数: do_sample=False, num_beams=3, temperature=1.0")
    print("特点: 维护多条候选路径，选择总概率最高的序列，质量更高")

    with torch.no_grad():
        beam_output = model.generate_super(
            x,
            stop_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            temperature=1.0,
            do_sample=False,
            num_beams=3
        )
        beam_response = tokenizer.decode(beam_output[0].tolist(), skip_special_tokens=True)

    print(f"束搜索结果: {beam_response}")
    print()

    # 测试4: 不同的温度参数对随机采样的影响
    print("🌡️ 测试4: 不同温度参数对随机采样的影响")
    print("参数: do_sample=True, num_beams=1, 测试不同temperature值")

    temperatures = [0.2, 0.8, 1.5]
    for temp in temperatures:
        with torch.no_grad():
            temp_output = model.generate_super(
                x,
                stop_id=tokenizer.eos_token_id,
                max_new_tokens=30,
                temperature=temp,
                do_sample=True,
                num_beams=1
            )
            temp_response = tokenizer.decode(temp_output[0].tolist(), skip_special_tokens=True)
            print(f"温度 {temp}: {temp_response}")

    print()
    print("=" * 60)
    print("✅ 三种解码策略测试完成！")
    print()
    print("📊 总结对比:")
    print("• 贪婪解码: 速度快，结果确定，适合确定性任务")
    print("• 随机采样: 创造性强，结果多样，适合创意生成")
    print("• 束搜索: 质量较高，平衡速度和质量，适合一般对话")

def test_original_generation():
    """
    原始的生成代码作为对比
    """
    model_id = "../model/kmno4zx/happy-llm-215M-sft/"

    print("运行原始生成代码...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cpu").eval()

    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]

    input_ids = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    input_ids = tokenizer(input_ids).data['input_ids']

    x = (torch.tensor(input_ids, dtype=torch.long)[None, ...]).to(model.device)

    with torch.no_grad():
        y = model.generate_super(x, stop_id=tokenizer.eos_token_id, max_new_tokens=512, temperature=0.6)
        response = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)

    print(f"Assistant: {response}")

if __name__ == "__main__":
    print("开始测试三种解码策略...")
    print()

    try:
        test_decoding_strategies()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("运行原始生成代码...")
        test_original_generation()