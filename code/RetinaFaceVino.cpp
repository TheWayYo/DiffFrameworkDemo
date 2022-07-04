#include "RetinaFaceVino.h"

const float mean_vals[3] = { 104.0f, 117.0f, 123.0f };
const float norm_vals[3] = { 1.0f, 1.0f, 1.0f };


RetinaFaceVino::RetinaFaceVino(const std::string& model_path) {
	try {
		retina_ = core.compile_model(model_path + "\\retina_sim.xml", "CPU");
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	infer_request_ = retina_.create_infer_request();

	//
	process->GenetatorAnchors(input_width, input_height, anchors);

}

RetinaFaceVino::~RetinaFaceVino() {

}



void RetinaFaceVino::DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceBoxes) {
	cv::Mat pad_image = process->PreProcess(image);
	int img_w = image.cols;
	int img_h = image.rows;
	int pad_img_w = pad_image.cols;
	int pad_img_h = pad_image.rows;

	cv::Mat input_image;
	cv::resize(pad_image, input_image, cv::Size(input_width, input_height));
	input_image.convertTo(input_image, CV_32F, 1.0);
	cv::subtract(input_image, cv::Scalar(104.0f, 117.0f, 123.0f), input_image);
	//cv::divide();

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

	int face_num = 0;
	faceBoxes.clear();
	int rst = process->PostProcess(anchors, (float*)tensor_bbox.data<const float>(), (float*)tensor_cls.data<const float>(), (float*)tensor_lm.data<const float>(), input_width, input_height, pad_img_w, pad_img_h, img_w, img_h, faceBoxes, 0.6f);
}

