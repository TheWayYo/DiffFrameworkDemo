#include "RetinaFaceNcnn.h"

const float mean_vals[3] = { 104.0f, 117.0f, 123.0f };
const float norm_vals[3] = { 1.0f, 1.0f, 1.0f };


RetinaFaceNcnn::RetinaFaceNcnn(const std::string& model_path) {
	std::string param_path = model_path + "/retina.param";
	std::string bin_path = model_path + "/retina.bin";

	retina_.load_param(param_path.data());
	retina_.load_model(bin_path.data());

	//
	process->GenetatorAnchors(input_width, input_height, anchors);
}

RetinaFaceNcnn::~RetinaFaceNcnn() {

}



void RetinaFaceNcnn::DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceBoxes) {
	cv::Mat pad_image = process->PreProcess(image);
	int img_w = image.cols;
	int img_h = image.rows;
	int pad_img_w = pad_image.cols;
	int pad_img_h = pad_image.rows;

	ncnn::Mat net_in = ncnn::Mat::from_pixels_resize(pad_image.data, ncnn::Mat::PIXEL_BGR, pad_img_w, pad_img_h, input_width, input_height);
	net_in.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = retina_.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(thread_num);
	ex.input("input0", net_in);

	ncnn::Mat tensor_bbox;
	ncnn::Mat tensor_lm;
	ncnn::Mat tensor_cls;
	ex.extract("586", tensor_cls);
	ex.extract("585", tensor_lm);
	ex.extract("output0", tensor_bbox);

	int face_num = 0;
	faceBoxes.clear();
	int rst = process->PostProcess(anchors, (float*)tensor_bbox.data, (float*)tensor_cls.data, (float*)tensor_lm.data, input_width, input_height, pad_img_w, pad_img_h, img_w, img_h, faceBoxes, 0.6f);
}


