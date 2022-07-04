#include "RetinaFaceTnn.h"


RetinaFaceTnn::RetinaFaceTnn(const std::string& model_path) {

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

	//
	process->GenetatorAnchors(input_width, input_height, anchors);
}

RetinaFaceTnn::~RetinaFaceTnn() {

}



void RetinaFaceTnn::DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceBoxes) {
	cv::Mat pad_image = process->PreProcess(image);
	int img_w = image.cols;
	int img_h = image.rows;
	int pad_img_w = pad_image.cols;
	int pad_img_h = pad_image.rows;

	cv::Mat input_image;
	cv::resize(pad_image, input_image, cv::Size(input_width, input_height));
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
	faceBoxes.clear();
	int rst = process->PostProcess(anchors, (float*)tensor_bbox->GetData(), (float*)tensor_cls->GetData(), (float*)tensor_lm->GetData(), input_width, input_height, pad_img_w, pad_img_h, img_w, img_h, faceBoxes, 0.6f);
}

std::string RetinaFaceTnn::content_buffer_from(const char* proto_or_model_path)
{
	std::ifstream file(proto_or_model_path, std::ios::binary);
	if (file.is_open())
	{
		file.seekg(0, file.end);
		int size = file.tellg();
		char* content = new char[size];
		file.seekg(0, file.beg);
		file.read(content, size);
		std::string file_content;
		file_content.assign(content, size);
		delete[] content;
		file.close();
		return file_content;
	} // empty buffer
	else
	{
		std::cout << "Can not open " << proto_or_model_path << "\n";
	}
}

void RetinaFaceTnn::transform(const cv::Mat& mat_rs)
{
	// be carefully, no deepcopy inside this tnn::Mat constructor,
	// so, we can not pass a local cv::Mat to this constructor.
	// push into input_mat
	auto input_shape = get_input_shape(instance, "input0");

	input_mat = std::make_shared<tnn::Mat>(tnn::DEVICE_X86, tnn::N8UC3,
		input_shape, (void*)mat_rs.data);
	if (!input_mat->GetData())
	{
		std::cout << "input_mat == nullptr! transform failed\n";
	}
}

tnn::DimsVector RetinaFaceTnn::get_input_shape(
	const std::shared_ptr<tnn::Instance>& _instance,
	std::string name)
{
	tnn::DimsVector shape = {};
	tnn::BlobMap blob_map = {};
	if (_instance)
	{
		_instance->GetAllInputBlobs(blob_map);
	}

	if (name == "" && blob_map.size() > 0)
		if (blob_map.begin()->second)
			shape = blob_map.begin()->second->GetBlobDesc().dims;

	if (blob_map.find(name) != blob_map.end()
		&& blob_map[name])
	{
		shape = blob_map[name]->GetBlobDesc().dims;
	}

	return shape;
}


