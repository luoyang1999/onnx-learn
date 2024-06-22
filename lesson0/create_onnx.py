import onnx 
from onnx import helper 
from onnx import TensorProto 
 
# ValueInfoProto
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])

# NodeProto make_node: [ops, input, outputs]
mul = helper.make_node('Mul', ['a', 'x'], ['c'])
add = helper.make_node('Add', ['c', 'b'], ['output'])

# GraphProto
graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])

# ModelProto
model = helper.make_model(graph) 

# Check the model
onnx.checker.check_model(model)

print(model)

graph = model.graph 
node = graph.node 
input = graph.input 
output = graph.output 
print(node) 
print(input) 
print(output) 

model.ir_version = 8
model.opset_import[0].version = 18
onnx.save(model, './linear_func.onnx')


