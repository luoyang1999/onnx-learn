import numpy as np
import torch
import time
import onnxruntime

MODEL_FILE = '.model.onnx'
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0
DEVICE=f'{DEVICE_NAME}:{DEVICE_INDEX}'

# A simple model to calculate addition of two tensors
def model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x, y):
            return x.add(y)

    return Model()

# Create an instance of the model and export it to ONNX graph format
def create_model(type: torch.dtype = torch.float32):
    sample_x = torch.ones(3, dtype=type)
    sample_y = torch.zeros(3, dtype=type)
    torch.onnx.export(model(), (sample_x, sample_y), MODEL_FILE,
                    input_names=["x", "y"], output_names=["z"], 
                    dynamic_axes={"x":{0 : "array_length_x"}, "y":{0: "array_length_y"}})

# Create an ONNX Runtime session with the provided model
def create_session(model: str) -> onnxruntime.InferenceSession:
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    return onnxruntime.InferenceSession(model, providers=providers)

# Run the model on CPU consuming and producing numpy arrays 
def run(x: np.array, y: np.array) -> np.array:
    session = create_session(MODEL_FILE)
    z = session.run(["z"], {"x": x, "y": y})
    return z[0]   

# Run the model on device consuming and producing ORTValues
def run_with_data_on_device(x: np.array, y: np.array) -> onnxruntime.OrtValue:
    session = create_session(MODEL_FILE)

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, DEVICE_NAME, DEVICE_INDEX)
    y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(y, DEVICE_NAME, DEVICE_INDEX)

    io_binding = session.io_binding()
    io_binding.bind_input(name='x', device_type=x_ortvalue.device_name(), device_id=0, element_type=x.dtype, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
    io_binding.bind_input(name='y', device_type=y_ortvalue.device_name(), device_id=0, element_type=y.dtype, shape=y_ortvalue.shape(), buffer_ptr=y_ortvalue.data_ptr())
    io_binding.bind_output(name='z', device_type=DEVICE_NAME, device_id=DEVICE_INDEX, element_type=x.dtype, shape=x_ortvalue.shape())
    session.run_with_iobinding(io_binding)

    z = io_binding.get_outputs()

    return z[0]

def main():
    create_model()
    # print(run(x=np.float32([1.0, 2.0, 3.0]),y=np.float32([4.0, 5.0, 6.0])))
    
    t1 = time.time()
    print(run(x=np.float32([1.0, 2.0, 3.0]),y=np.float32([4.0, 5.0, 6.0])))
    # [array([5., 7., 9.], dtype=float32)]t1 = time.time()
    t2 = time.time()

    print(run_with_data_on_device(x=np.float32([1.0, 2.0, 3.0, 4.0, 5.0]), y=np.float32([1.0, 2.0, 3.0, 4.0, 5.0])).numpy())
    # [ 2.  4.  6.  8. 10.]
    t3 = time.time()
    
    print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference.')
    print(f'Done. ({(1E3 * (t3 - t2)):.1f}ms) Inference.')

if __name__ == "__main__":
    main()   
