from srcnn import init_torch_model
import onnx
import cv2
import onnxruntime
import numpy as np
import torch
import time


# 对模型运行速度进行比较

# 方式1，直接推理
model_org = init_torch_model()
# 方式2，onnx
model_onnx = onnx.load("lesson1/srcnn.onnx") 
 
# HWC to NCHW 
input_img = cv2.imread('lesson1/face.png').astype(np.float32) 
input_img = np.transpose(input_img, [2, 0, 1]) 
input_img = np.expand_dims(input_img, 0)

test_time = 10

org_beign = time.time()
for _ in range(test_time):
    org_output = model_org(torch.Tensor(input_img))
org_end = time.time()

ort_beign = time.time()
ort_session = onnxruntime.InferenceSession("lesson1/srcnn.onnx") 
for _ in range(test_time):
    ort_inputs = {'input': input_img} 
    ort_output = ort_session.run(['output'], ort_inputs)[0]
ort_end = time.time()

print(org_end - org_beign, ort_end - ort_beign)

