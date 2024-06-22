import torch


print(torch.version.cuda)

# 检查是否有CUDA支持
if torch.cuda.is_available():
    # 获取GPU的数量
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        # 获取并打印GPU名称
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i} Name: {device_name}")
else:
    print("CUDA is not available.")