import onnx 
 
onnx_model = onnx.load("lesson1/srcnn.onnx") 
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")