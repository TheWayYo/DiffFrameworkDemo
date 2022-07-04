



# TVM在windows环境配置

## 环境要求：

VS2019，64bit

cmake-gui

python 64bit（决定llvm只能是64位，tvm也只能是64位）

## 安装zlib

* 下载地址：http://www.zlib.net/

* 用cmake-gui编译：

![2022619-101830](F:\work\learn\projects\DiffModelInfer\images\2022619-101830.jpg)

* 用VS打开build目录下的.sln文件

![2022619-102342](F:\work\learn\projects\DiffModelInfer\images\2022619-102342.jpg)

修改编译的模式为release，选中ALL_BUILD，右击，build即可。

* 编译完成后，对应文件夹build\Release中生成：lib和dll等文件

## 安装LLVM

* 下载地址：https://codeload.github.com/llvm/llvm-project/zip/refs/tags/llvmorg-11.0.1

* 解压后进入到llvm-project-llvmorg-11.0.1\llvm，并新建一个build文件夹
* cd build中
* 在power shell或cmd中执行：

为了避免后面使用tmvc出现错误

RuntimeError: Can not find the LLVM clang for Windows clang.exe， 建议加上：-DLLVM_ENABLE_PROJECTS=clang

```bash
X64:
cmake -Thost=x64 -DLLVM_ENABLE_PROJECTS=clang ..
win32：
cmake -A win32 -DLLVM_ENABLE_PROJECTS=clang ..
```

* 执行

```bash
cd ..
cmake --build build --config Release -- /m
```

然后就是一个漫长的等待：

如果遇到了堆栈溢出的问题，

“引发类型为“System.OutOfMemoryException”的异常。”

则在build目录下执行：

```bash
D:\software\vs2019\MSBuild\Current\Bin\amd64\MSBuild.exe LLVM.sln /property:Configuration=Release
```

* 安装完后

在build目录下，会有一个Release，bin和lib文件夹，里面保存了对应的exe和lib文件

* 配置环境变量

![2022619-120506](F:\work\learn\projects\DiffModelInfer\images\2022619-120506.jpg)

```bash
llvm-project-llvmorg-12.0.1\llvm\build\Release\bin 你的目录
```

* 重启电脑

## 安装TVM

* 下载：

链接：https://pan.baidu.com/s/1KKNw_qE49gcqVn-Ihczphw 
提取码：pw5h

最新版，我这边编译的时候报了一个错误。

* 新建一个build文件夹

* 解压并修改cmake/config.cmake中 `set(USE_LLVM OFF)`，改为：

```bash
set(USE_LLVM  F:/work/share/TVM/llvm-project-llvmorg-11.0.1/llvm/build/Release/bin/llvm-config.exe)
```

注意要使用“/”

* 将config.cmake拷贝到build目录下，并执行

```bash
x64：
cmake -A x64 -Thost=x64 ..
win32：
cmake -A win32 ..
```

* 执行

```bash
cd ..
cmake --build build --config Release -- /m
```

* 成功，则会在build/Release下有：

tvm.lib、tvm_runtime.lib等文件

* 如果报错z.lib找不到等问题

则使用VS打开build中的tvm.sln，并且将zlib的lib路径

```
F:\work\share\TVM\zlib1212\zlib-1.2.12\build\Release
```

添加到Properties->Configureation Properties->Linker->General->Additional Library Directories中。

并修改Properties->Configureation Properties->Linker->Input->Additional Dependencies的z.lib改为zlib.lib

最后，重新生成tvm即可

* 执行

```bash
cd python 
python setup.py install
```

* 安装完成执行测试，没报错则完成安装

```
tvmc -h
```

* 测试，转换成功

```bash
tvmc compile --target llvm .\retina.onnx -o .\retina.tar
```

## 使用代码进行模型转换

retina.onnx的获取，通过以下开源工程：

https://github.com/biubug6/Pytorch_Retinaface

使用代码进行模型转换：

```python
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
loaded_lib = tvm.runtime.load_module(libpath) #老的版本使用tvm.module.load
loaded_params = bytearray(open(param_path, "rb").read())

# 这里执行的平台为CPU
ctx = tvm.cpu()

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.set_input("input0", x)
module.run()
out_deploy = module.get_output(0).asnumpy()

print(out_deploy)
```



# TVM模型推理

## python版本

```python
import tvm
import numpy as np
from tvm.contrib import graph_runtime
import cv2
import torch
from utils import decode, decode_landm, PriorBox, nms

cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 320,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

image_path = "F:\\work\\learn\\projects\\demo\\RetinaFace\\x64\\Release\\test.jpeg"
graph_json_path = "./models/retina.json"
libpath = "./models/retina.so"
param_path = "./models/retina.params"
vis_thres = 0.1
confidence_threshold = 0.02
nms_threshold = 0.4
top_k = 5000
keep_top_k = 100

def detect():
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #补边
    size = max(img_raw.shape[0], img_raw.shape[1])
    new_img = np.zeros((size, size, 3))
    new_img[:img_raw.shape[0], :img_raw.shape[1], :] = img_raw
    img = cv2.resize(new_img, (320, 320))
    im_height, im_width, _ = img.shape
    #缩放比例
    scale_x = new_img.shape[1] / im_width
    scale_y = new_img.shape[0] / im_height
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    img = np.float32(img)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    # 加载模型
    loaded_json = open(graph_json_path).read()
    loaded_lib = tvm.runtime.load_module(libpath) #老的版本使用tvm.module.load
    loaded_params = bytearray(open(param_path, "rb").read())

    # 执行模型推理
    ctx = tvm.cpu()
    device = torch.device("cpu")
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("input0", img)
    module.run()
    bbox_tensor = module.get_output(0).asnumpy()
    classify_tensor = module.get_output(1).asnumpy()
    landmark_tensor = module.get_output(2).asnumpy()

    #后处理
    bbox_tensor = torch.from_numpy(bbox_tensor)
    landmark_tensor = torch.from_numpy(landmark_tensor)
    classify_tensor = torch.from_numpy(classify_tensor)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(torch.Tensor(bbox_tensor).squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = classify_tensor.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landmark_tensor.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        cv2.rectangle(img_raw, (int(b[0] * scale_x), int(b[1] * scale_y)), (int(b[2] * scale_x), int(b[3] * scale_y)), (0, 0, 255), 2)

        # landms
        cv2.circle(img_raw, (int(b[5]*scale_x), int(b[6]*scale_y)), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (int(b[7]*scale_x), int(b[8]*scale_y)), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (int(b[9]*scale_x), int(b[10]*scale_y)), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (int(b[11]*scale_x), int(b[12]*scale_y)), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (int(b[13]*scale_x), int(b[14]*scale_y)), 1, (255, 0, 0), 4)
    # save image

    name = "test_res.jpg"
    cv2.imwrite(name, img_raw)


if __name__ == '__main__':
    detect()
```



## C++版本

VS环境配置：

添加include：

Properties->Configureation Properties->VC++ Directories->Include Directories 中添加 opencv的头文件include路径，tvm的include，3rdparty\dlpack\include，3rdparty\dmlc-core\include

添加Library路径：

Properties->Configureation Properties->VC++ Directories->Library Directories中添加opencv的lib文件路径，TVM的lib文件夹目录

添加lib文件：

Properties->Configureation Properties->Linker->Input->Additional Dependencies中添加opencv_world3414.lib;tvm_runtime.lib

配置：

Properties->Configureation Properties->C/C++->Preprocessor->Preprocessor Definitions中添加：_CRT_SECURE_NO_WARNINGS

**模型初始化部分**

```c++
DLDevice dev{ kDLCPU, 0 };
tvm::runtime::Module retina_model;
tvm::runtime::PackedFunc set_input;
tvm::runtime::PackedFunc get_output;
tvm::runtime::PackedFunc load_params;
tvm::runtime::PackedFunc run;

retina_model = tvm::runtime::Module::LoadFromFile(model_path + "\\retina.so");
	
//json graph
std::fstream json_in(model_path + "\\retina.json", std::ios::in);
std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
json_in.close();
//parameters
std::ifstream params_in(model_path + "\\retina.params", std::ios::binary);
std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
params_in.close();
TVMByteArray params_arr;
params_arr.data = params_data.c_str();
params_arr.size = params_data.length();

int device_type = kDLCPU;

//"tvm.graph_runtime.create"老版本使用
tvm::runtime::Module gmod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, retina_model, device_type, 0);
set_input = gmod.GetFunction("set_input");
get_output = gmod.GetFunction("get_output");
load_params = gmod.GetFunction("load_params");
run = gmod.GetFunction("run");
load_params(params_arr);
```

**模型推理部分**

```c++
//输入
tvm::runtime::NDArray input_tensor = tvm::runtime::NDArray::Empty({ 1, input_image.channels(), input_height, input_width}, DLDataType{ kDLFloat, 32, 1 }, dev);
//输出
tvm::runtime::NDArray bbox_tensor = tvm::runtime::NDArray::Empty({ 1, 4200, 4 }, DLDataType{ kDLFloat, 32, 1 }, dev);
tvm::runtime::NDArray landmark_tensor = tvm::runtime::NDArray::Empty({ 1, 4200, 10 }, DLDataType{ kDLFloat, 32, 1 }, dev);
tvm::runtime::NDArray classify_tensor = tvm::runtime::NDArray::Empty({ 1, 4200, 2 }, DLDataType{ kDLFloat, 32, 1 }, dev);

float* input_data = static_cast<float*>(input_tensor->data);
size_t image_size = input_width * input_height;
for (size_t row = 0; row < input_height; row++)
{
    for (size_t col = 0; col < input_width; col++)
    {
        for (size_t c = 0; c < input_image.channels(); c++)
        {
            input_data[image_size * c + row * input_width + col] = input_image.at<cv::Vec3f>(row, col)[c];
        }
    }
}
//设置输入
set_input("input0", input_tensor);
run();
//获取输出
get_output(0, bbox_tensor);
get_output(1, classify_tensor);
get_output(2, landmark_tensor);
```



