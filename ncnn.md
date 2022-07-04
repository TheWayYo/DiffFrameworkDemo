# ncnn配置与使用

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
* 下载gmock源码，解压后，重命名为gmock，放在protobuf目录下即可。（没有报错则不用处理）
* 取消protobuf_BUILD_TESTS, 
* 勾选protobuf_BUILD_SHARED_LIBS,
* 勾选ZLIB，并指定ZLIB路径（重要）ZLIB_LIBRARY ZLIB_INCLUDE_DIR
* 编译后，生成对应的链接库。
* 编译Install项目，
* 在系统的环境变量中，在系统变量PATH中添加路径：D:\protobuf-3.4.0\Builds\install\bin；然后重启系统！ 

## 安装opencv

* 下载： https://opencv.org/
* 按提示安装即可

## 安装NCNN

* 下载ncnn: https://github.com/Tencent/ncnn
* 打开CMake-GUI：  设置 源文件目录：D:/ncnn-master  设置 目标文件目录：D:/ncnn-master/Builds  点击 Configure
* 点击add entry，根据自己的路径加入以下内容：

```cpp
Protobuf_LIBRARIES=D:\protobuf-3.4.0\Builds\install\lib\libprotobuf.lib
Protobuf_INCLUDE_DIR=D:\protobuf-3.4.0\Builds\install\include
Protobuf_PROTOC_EXECUTABLE=D:\protobuf-3.4.0\Builds\install\install/bin/protoc.exe
CMAKE_CONFIGURATION_TYPES=Release
Protobuf_SRC_ROOT_FOLDER=D:/protobuf-3.4.0/src
OpenCV_DIR=.../lib （可以不要，OPenCV用于example，没有用取消勾选ncnn_build_example）
OpenCV_INCLUDE_DIRS
```
* 再次点击 Configure  点击 Generate
* 打开 目标文件目录（D:/ncnn-master/Builds）中的工程文件（ncnn.sln），编译其中的INSTALL项目即可
* 若要生成转换模型的工具onnx2ncnn和caffe2ncnn则：

Properties->Configureation Properties->C/C++->Preprocessor->Preprocessor Definitions中添加：PROTOBUF_USE_DLLS，再重新build

## 模型转换

需要把libprotobuf.dll放到onnx2ncnn目录下

```bash
pip install onnx-simplifier
python -m onnxsim retina.onnx retina_sim.onnx
F:\work\share\NCNN\ncnn\build\install\bin\onnx2ncnn.exe retina_sim.onnx retina.param retina.bin
```

## NCNN使用

Properties->Configureation Properties->VC++ Directories->Include Directories中添加：

```
ncnn\build\install\include
```

Properties->Configureation Properties->VC++ Directories->Library Directories中添加：

```
ncnn\build\install\lib
```

Properties->Configureation Properties->Linker->Input->Additional Dependencies：

```
ncnn.lib
```

将dll添加到exe对应文件夹中：

```
ncnn.dll
```

模型加载

```c++
std::string param_path = model_path + "/retina.param";
std::string bin_path = model_path + "/retina.bin";

retina_.load_param(param_path.data());
retina_.load_model(bin_path.data());
```

模型推理

```c++
ncnn::Mat net_in = ncnn::Mat::from_pixels_resize(pad_image.data, ncnn::Mat::PIXEL_BGR, pad_img_w, pad_img_h, input_width, input_height);
net_in.substract_mean_normalize(mean_vals, norm_vals);
ncnn::Extractor ex = retina_.create_extractor();
ex.set_light_mode(true);
ex.set_num_threads(thread_num);
ex.input("input0", net_in);

ncnn::Mat tensor_bbox_;
ncnn::Mat tensor_lm_;
ncnn::Mat tensor_cls_;
ex.extract("586", tensor_cls_);
ex.extract("585", tensor_lm_);
ex.extract("output0", tensor_bbox_);
```

