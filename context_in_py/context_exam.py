class MyContextManager:
    def __enter__(self):
        print("进入上下文")
        return self  # 返回的对象会被赋值给as后面的变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出上下文")
        # 返回True表示异常已经被处理
        # 返回False或None表示异常需要继续传播
        return False


# 使用示例
with MyContextManager() as cm:
    print("在上下文中执行操作")
#     进入上下文
# 在上下文中执行操作
# 退出上下文
