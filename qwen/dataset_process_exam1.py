while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = "**"
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答