
# 加载权重先cpoy_在del删除引用，及时释放内存

```python
import torch.cuda

param.cpoy_(param_data)
del param_data
# 如果需要，还有释放GPU内存
torch.cuda.empty_cache()
```

## 原始加载权重方法

```python
# Then load pretrained weights

start_memory_tracking()

model = GPTModel(BASE_CONFIG)
model.to(device)

model.load_state_dict(
    torch.load("model.pth", map_location=device, weights_only=True)
)
model.to(device)
model.eval();

print_memory_usage()
# Maximum GPU memory allocated: 12.8 GB
```

## 1. 先加载权重都cpu上，在复制到模型中

```python
start_memory_tracking()

model = GPTModel(BASE_CONFIG).to(device)

state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

print_memory_usage()

# Sequentially copy weights to the model's parameters
with torch.no_grad():
    for name, param in model.named_parameters():
        if name in state_dict:
            param.copy_(state_dict[name].to(device))
        else:
            print(f"Warning: {name} not found in state_dict.")

print_memory_usage()
# Maximum GPU memory allocated: 6.4 GB
# Maximum GPU memory allocated: 6.7 GB
```

## 2. 使用meta 是和cpu内存小， GPU内存大的

```python
import os
import psutil
from threading import Thread


def memory_usage_in_gb(func, *args, **kwargs):
    process = psutil.Process(os.getpid())

    # Measure the baseline memory usage before running the function
    baseline_mem = process.memory_info().rss / 1024 ** 3  # in GB

    # Start monitoring memory in a separate thread
    mem_usage = []
    done = False

    def monitor_memory():
        while not done:
            mem_usage.append(process.memory_info().rss / 1024 ** 3)  # Convert to GB
            time.sleep(0.1)

    t = Thread(target=monitor_memory)
    t.start()

    # Run the function
    func(*args, **kwargs)

    # Stop monitoring
    done = True
    t.join()

    peak_mem_usage_gb = max(mem_usage) - baseline_mem
    return peak_mem_usage_gb
```

### 顺序加载的CPU内存消耗，先载入cpu中在赋值权重到GPU
```python
def load_sequentially():
    start_memory_tracking()

    model = GPTModel(BASE_CONFIG).to(device)

    state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

    print_memory_usage()

    # Sequentially copy weights to the model's parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name].to(device))
            else:
                print(f"Warning: {name} not found in state_dict.")

    print_memory_usage()


peak_memory_used = memory_usage_in_gb(load_sequentially)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
# Maximum GPU memory allocated: 6.4 GB
# Maximum GPU memory allocated: 6.7 GB
# -> Maximum CPU memory allocated: 6.3 GB
```

### 使用device=meta来加载权重

```python
def load_sequentially_with_meta():
    start_memory_tracking()

    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)

    model = model.to_empty(device=device)

    state_dict = torch.load("model.pth", map_location=device, weights_only=True)

    print_memory_usage()

    # Sequentially copy weights to the model's parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
            else:
                print(f"Warning: {name} not found in state_dict.")

    print_memory_usage()

peak_memory_used = memory_usage_in_gb(load_sequentially_with_meta)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
# Maximum GPU memory allocated: 12.8 GB
# Maximum GPU memory allocated: 12.8 GB
# -> Maximum CPU memory allocated: 1.3 GB
```

### 直接使用torch.load 加载权重

```python
def baseline():
    start_memory_tracking()

    model = GPTModel(BASE_CONFIG)
    model.to(device)

    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval();

    print_memory_usage()

peak_memory_used = memory_usage_in_gb(baseline)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
# Maximum GPU memory allocated: 12.8 GB
# -> Maximum CPU memory allocated: 4.4 GB
```
### 使用`mmap=True`（一般配以`map_location='cpu/meta`）

```python
def best_practices():
  with torch.device("meta"):
      model = GPTModel(BASE_CONFIG)

  model.load_state_dict(
      torch.load("model.pth", map_location=device, weights_only=True, mmap=True),
      assign=True
  )

  print_memory_usage()

peak_memory_used = memory_usage_in_gb(best_practices)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
# Maximum GPU memory allocated: 6.4 GB
# -> Maximum CPU memory allocated: 5.9 GB
```

### 分参数保存，分别读取

```python
model = GPTModel(BASE_CONFIG)
# Assume `model` is your trained model
state_dict = model.state_dict()

# Create a directory to store individual parameter files
os.makedirs("model_parameters", exist_ok=True)

# Save each parameter tensor separately
for name, param in state_dict.items():
    torch.save(param.cpu(), f"model_parameters/{name}.pt")

del model

def load_individual_weights():

    start_memory_tracking()

    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)

    model = model.to_empty(device=device)

    print_memory_usage()
    param_dir = "model_parameters"

    with torch.no_grad():
        for name, param in model.named_parameters():
            weight_path = os.path.join(param_dir, f"{name}.pt")
            if os.path.exists(weight_path):
                param_data = torch.load(weight_path, map_location="cpu", weights_only=True)
                param.copy_(param_data)
                del param_data  # Free memory
            else:
                print(f"Warning: {name} not found in {param_dir}.")

    print_memory_usage()


peak_memory_used = memory_usage_in_gb(load_individual_weights)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
# Maximum GPU memory allocated: 6.4 GB
# Maximum GPU memory allocated: 6.4 GB
# -> Maximum CPU memory allocated: 0.3 GB
```










