import onnx
import cv2
from tvm.contrib import graph_runtime
import tvm
import numpy as np
from tvm import relay

# 开始同样是读取.onnx模型
onnx_model = onnx.load('./retina.onnx')
img = cv2.imread("F:\\work\\learn\\projects\\demo\\RetinaFace\\x64\\Release\\test.jpeg")
img = cv2.resize(img, (320, 320))

# 以下的图片读取仅仅是为了测试
img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img / 255.0  # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

# 这里首先在PC的CPU上进行测试 所以使用LLVM进行导出
target = tvm.target.create('llvm')

input_name = 'input0'  # change '1' to '0'
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# 这里利用TVM构建出优化后模型的信息
with relay.build_config(opt_level=2):
    graph, lib, params = relay.build_module.build(sym, target, params=params)

dtype = 'float32'

# 下面的函数导出我们需要的动态链接库 地址可以自己定义
print("Output model files")
libpath = "./models/retina.so"
lib.export_library(libpath)

# 下面的函数导出我们神经网络的结构，使用json文件保存
graph_json_path = "./models/retina.json"
with open(graph_json_path, 'w') as fo:
    fo.write(graph)

# 下面的函数中我们导出神经网络模型的权重参数
param_path = "./models/retina.params"
with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))

# 接下来我们加载导出的模型去测试导出的模型是否可以正常工作
loaded_json = open(graph_json_path).read()
loaded_lib = tvm.runtime.load_module(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

# 这里执行的平台为CPU
ctx = tvm.cpu()

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.set_input("input0", x)
module.run()
out_deploy = module.get_output(0).asnumpy()

print(out_deploy)