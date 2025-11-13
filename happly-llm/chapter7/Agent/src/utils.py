import inspect

def function_to_json(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        list: "array",
        bool: "boolean",
        dict: "object",
        type(None): "null",
    }

    # 获取函数签名
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise  ValueError(
            f'无法获取函数 {func.__name__} 的签名: {str(e)}'
        )
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(f'未知的类型注解 {param.annotation}, 参数名 {param.name}: {str(e)}')
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "", # 函数的文档支付宝(不存在就是空字符串)
            "parameters":{
                "type": "object",
                "properties": parameters,
                "required": required # 必须参数的列表
            },
        },
    }

