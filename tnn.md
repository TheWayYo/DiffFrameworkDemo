# TNN在windows的环境配置

## 安装zlib

* 下载链接：http://www.zlib.net/
* 打开cmake-gui
* 配置好源文件路径和build路径
* 建议生成静态库lib，有一个选项开关
* 可以修改编译的库类型和安装路径prefix，configure和genrate
* 打开sln文件
* 生成ALL_BUILD和INSTALL

## 安装protobuf

* 下载地址：https://github.com/protocolbuffers/protobuf/releases?page=8
* 源目录选择cmakelis文件所在目录，选择protobuf-3.4.0文件夹中的cmake文件夹
* 点击Configure，提示找不到gmock目录。
* 下载gmock源码，解压后，重命名为gmock，放在protobuf目录下即可。（没有报错则不用处理）（3.4.0版本才需要）
* 取消protobuf_BUILD_TESTS, 
* 勾选protobuf_BUILD_SHARED_LIBS,
* 勾选ZLIB，并指定ZLIB路径（重要）ZLIB_LIBRARY ZLIB_INCLUDE_DIR
* 编译后，生成对应的链接库。
* 编译Install项目，
* 在系统的环境变量中，在系统变量PATH中添加路径：D:\protobuf-3.4.0\Builds\install\bin；然后重启系统！（不配置以后使用就需要手动添加）

```
set TNN_DIR=%~dp0..\

@echo off
echo %TNN_DIR%
echo %1

if "%2" == "" (
    goto init_fold
) else (
    goto init_env
)

:init_env
    if %1 == x86 (
        echo "build x86"
        call "D:/software/vs2019/VC/Auxiliary/Build/vcvars32.bat"
    ) else (
        echo "build x64"
        call "D:/software/vs2019/VC/Auxiliary/Build/vcvars64.bat"
    )
    goto init_fold

:init_fold
    mkdir build_win
    cd build_win

cmake %TNN_DIR% ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_X86_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ^ 
-DTNN_CONVERTER_ENABLE=ON ^
-DTNN_ONNX2TNN_ENABLE=ON ..

cmake --build . --config Release

```

## 安装TNN

方式1：不需要安装zlib和protobuf

修改scripts\build_msvc_native.bat，只能成功安装TNN.lib，不能成功安装onnx2tnn工具，安装之后onnx2tnn的工具没有成功

```bash
set TNN_DIR=%~dp0..\

@echo off
echo %TNN_DIR%
echo %1

if "%2" == "" (
    goto init_fold
) else (
    goto init_env
)

:init_env
    if %1 == x86 (
        echo "build x86"
        call "D:/software/vs2019/VC/Auxiliary/Build/vcvars32.bat"
    ) else (
        echo "build x64"
        call "D:/software/vs2019/VC/Auxiliary/Build/vcvars64.bat"
    )
    goto init_fold

:init_fold
    mkdir build_win
    cd build_win

cmake %TNN_DIR% ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_X86_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ..

cmake --build . --config Release

```

方式2：使用cmake-gui+VS编译失败

* Cmake-gui中添加源文件路径和build路径，点击configure
* 修改protobuf的路径

```bash
Protobuf_LIBRARIES=C:/Program Files/protobuf/lib/libprotobuf.lib
Protobuf_LIELIBRARY_RELEASE=C:/Program Files/protobuf/lib/libprotobuf-lite.lib
Protobuf_INCLUDE_DIR=C:/Program Files/protobuf/include
Protobuf_PROTOC_EXECUTABLE=C:/Program Files/protobuf/bin/protoc.exe
Protobuf_SRC_ROOT_FOLDER=F:/work/share/TNN/protobuf-3.6.1/src
```

* 勾选TNN_ONNX2TNN_ENABLE、TNN_CONVERTER_ENABLE
* configure, generate, open project,能够正常编译生成TNN，但onnx2tnn不能正常编译
* 修改onnx2tnn工程：

Properties->Configureation Properties->C/C++->Preprocessor->Command Line为

```bash
%(AdditionalOptions) /utf-8 -g -std=c++11
```

* 由于很多头文件引用的均是unix下的，所以尝试替换：

onnx2tnn_convert.cc中：

#include "unistd.h"变为：#include <io.h>    #include <process.h>

#include <sys/xattr.h>变为：。。。。。不成功

方式3：使用mingw

* 修改cmakelists.txt中加入

```bash
set(Protobuf_LIBRARIES "C:/Program Files/protobuf/lib/libprotobuf.lib")
set(Protobuf_LIELIBRARY_RELEASE "C:/Program Files/protobuf/lib/libprotobuf-lite.lib")
set(Protobuf_INCLUDE_DIR "C:/Program Files/protobuf/include")
set(Protobuf_PROTOC_EXECUTABLE "C:/Program Files/protobuf/bin/protoc.exe")
set(Protobuf_SRC_ROOT_FOLDER "F:/work/share/TNN/protobuf-3.6.1/src")
```

* 修改tools/convert2tnn/build.sh

## 模型推理

环境配置：

Properties->Configureation Properties->VC++ Directories->Include Directories中添加：

```bash
F:\work\share\TNN\TNN\include
```

Properties->Configureation Properties->VC++ Directories->Library Directories添加:

```
F:\work\share\TNN\TNN\scripts\build_win\Release
```

Properties->Configureation Properties->Linker->Input->Additional Dependencies中添加lib：

```c++
TNN.lib
```

将dll添加到exe对应文件夹中：

```
TNN.dll
```

模型加载

```c++
std::string proto_buffer, model_buffer;
proto_buffer = content_buffer_from((model_path + "//retina.tnnproto").data());
model_buffer = content_buffer_from((model_path + "//retina.tnnmodel").data());

TNN_NS::ModelConfig model_config;
model_config.model_type = tnn::MODEL_TYPE_TNN;
//proto file content saved to proto_buffer
model_config.params.push_back(proto_buffer);
//model file content saved to model_buffer
model_config.params.push_back(model_buffer);
retina->Init(model_config);

TNN_NS::NetworkConfig config;
config.device_type = tnn::DEVICE_X86;
config.library_path = { "" };
TNN_NS::Status error;
instance = retina->CreateInst(config, error);
instance->SetCpuNumThreads((int)thread_num);
```

模型推理

```c++
transform(input_image);

tnn::MatConvertParam input_cvt_param;
input_cvt_param.scale = scale_vals;
input_cvt_param.bias = bias_vals;

auto status = instance->SetInputMat(input_mat, input_cvt_param, "input0");

instance->Forward();

std::shared_ptr<tnn::Mat> tensor_cls;
std::shared_ptr<tnn::Mat> tensor_lm;
std::shared_ptr<tnn::Mat> tensor_bbox;
tnn::MatConvertParam cvt_param;

instance->GetOutputMat(tensor_cls, cvt_param, "586", tnn::DEVICE_X86);
instance->GetOutputMat(tensor_lm, cvt_param, "585", tnn::DEVICE_X86);
instance->GetOutputMat(tensor_bbox, cvt_param, "output0", tnn::DEVICE_X86);
```

