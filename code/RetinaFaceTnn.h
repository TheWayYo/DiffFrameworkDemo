#pragma once
#ifndef __RETINA_FACE_TNN_H_
#define __RETINA_FACE_TNN_H_

#include "Process.h"
#include <queue>
#include <memory>
#include <string>
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include <iostream>
#include <fstream>

class RetinaFaceTnn
{
public:
	RetinaFaceTnn(const std::string& model_path);

	~RetinaFaceTnn();

	void Init(int thread_num);

	void DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceboxes);

private:
	std::string content_buffer_from(const char* proto_or_model_path);

	void transform(const cv::Mat& mat_rs);

	tnn::DimsVector get_input_shape(
		const std::shared_ptr<tnn::Instance>& _instance,
		std::string name);

private:

	std::shared_ptr<tnn::TNN> retina = std::make_shared<tnn::TNN>();
	std::shared_ptr<tnn::Instance> instance;
	std::shared_ptr<tnn::Mat> input_mat;

	std::vector<float> scale_vals = { 1.0f, 1.0f, 1.0f };
	std::vector<float> bias_vals = { -104.0f, -117.0f, -123.0f };
	Process* process = new Process();
	int input_width = 400;
	int input_height = 400;
	int thread_num = 1;
	std::vector<std::vector<int>> anchors;

};
#endif

