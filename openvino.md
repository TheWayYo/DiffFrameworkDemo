

# Openvino环境配置

新版的openvino安装会简单很多

## 安装Visual Studio 2019

**1、离线下载**

VS地址：https://docs.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers

**2、创建文件夹用来存放安装包文件**

创建一个空的文件夹来下载vs2019离线安装包文件，将下载的引导程序放到此文件中

**3、下载c++桌面开发**

```c++
vs_community.exe --layout c:\localVScache --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --lang en-US
```

**4、打包安装**

执行vs_steup.exe

## 安装OpenVino Runtime

**方式1**

不用安装，直接下载release包，里面包含include、lib、bin使用的包

地址：

https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.1/

**方式2**

下载地址：

https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html

![image-20220529171505160](..\images\image-20220529171505160.png)

## 安装OpenVino Dev tools(Linux\windows)

* `python -m venv openvino_env`

* `openvino_env\Scripts\activate`
* `python -m pip install --upgrade pip`
* `pip install openvino-dev[tensorflow2,onnx]`
* `mo -h`

可以通过mo --input_model ""实现模型转换

## VS环境使用

Properties->Configureation Properties->VC++ Directories->Include Directories中添加：

```
openvino_2022.1.0.643\runtime\include
openvino_2022.1.0.643\runtime\include\ie
```

Properties->Configureation Properties->VC++ Directories->Library Directories中添加：

```
openvino_2022.1.0.643\runtime\3rdparty\tbb\lib
openvino_2022.1.0.643\runtime\lib\intel64\Release
```

Properties->Configureation Properties->Linker->Input->Additional Dependencies：

```
openvino.lib
```

将dll添加到exe对应文件夹中：

```
gna.dll
openvino.dll
openvino_auto_plugin.dll
openvino_intel_cpu_plugin.dll
openvino_onnx_frontend.dll
plugins.xml
tbb.dll
```

模型加载

```c++
try {
		retina_ = core.compile_model(model_path + "/retina.onnx", "CPU");
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	infer_request_ = retina_.create_infer_request();
```

模型推理

```c++
ov::Tensor input_tensor = infer_request_.get_input_tensor();
	ov::Shape input_shape = input_tensor.get_shape();
	size_t num_channels = input_shape[1];
	size_t w = input_shape[2];
	size_t h = input_shape[3];
	size_t image_size = h * w;

	float* input_data = input_tensor.data<float>();
	for (size_t row = 0; row < h; row++)
	{
		for (size_t col = 0; col < w; col++)
		{
			for (size_t c = 0; c < num_channels; c++)
			{
				input_data[image_size * c + row * w + col] = input_image.at<cv::Vec3f>(row, col)[c];
			}
		}
	}

	infer_request_.start_async();
	try {
		infer_request_.wait();
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	auto tensor_bbox = infer_request_.get_tensor("output0");
	auto tensor_lm = infer_request_.get_tensor("585");
	auto tensor_cls = infer_request_.get_tensor("586");
```

