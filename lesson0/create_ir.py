import torch
import torchvision

model = torchvision.models.resnet18()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# ir 生成
with torch.no_grad():
    # trace 指的是进行一次模型推理，在推理的过程中记录所有经过的计算，将这些记录整合成计算图
    jit_model = torch.jit.trace(model, dummy_input)

    print(jit_model.layer1.graph)
    print(jit_model.layer1.code)
