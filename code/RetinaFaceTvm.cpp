#include "RetinaFaceTvm.h"

const float mean_vals[3] = { 104.0f, 117.0f, 123.0f };
const float norm_vals[3] = { 1.0f, 1.0f, 1.0f };


RetinaFaceTvm::RetinaFaceTvm(const std::string& model_path) {
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
	
	//
	process->GenetatorAnchors(input_width, input_height, anchors);
}

RetinaFaceTvm::~RetinaFaceTvm() {

}



void RetinaFaceTvm::DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceBoxes) {
	//预处理
	cv::Mat pad_image = process->PreProcess(image);
	int img_w = image.cols;
	int img_h = image.rows;
	int pad_img_w = pad_image.cols;
	int pad_img_h = pad_image.rows;
	cv::Mat input_image;
	cv::resize(pad_image, input_image, cv::Size(input_width, input_height));
	input_image.convertTo(input_image, CV_32F, 1.0);
	cv::subtract(input_image, cv::Scalar(104.0f, 117.0f, 123.0f), input_image);
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

	set_input("input0", input_tensor);
	run();
	get_output(0, bbox_tensor);
	get_output(1, classify_tensor);
	get_output(2, landmark_tensor);
	

	int face_num = 0;
	std::vector<FaceBbox> faceboxes;
	int rst = process->PostProcess(anchors, (float*)bbox_tensor->data, (float*)classify_tensor->data, (float*)landmark_tensor->data, input_width, input_height, pad_img_w, pad_img_h, img_w, img_h, faceBoxes, 0.6f);
}


