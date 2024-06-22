import onnx  
 
onnx.utils.extract_model('./whole_model.onnx', 'partial_model.onnx', ['22'], ['28']) 