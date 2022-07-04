#pragma once
#ifndef __RETINA_FACE_H__
#define __RETINA_FACE_H__

#include <iostream>
#include <string>
#include "Process.h"
#include <openvino/openvino.hpp>
#include <queue>
#include <memory>

class RetinaFaceVino {
public:
	RetinaFaceVino(const std::string& model_path);

	~RetinaFaceVino();

	void Init(int thread_num);

	void DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceboxes);

	ov::Core core;
	ov::CompiledModel retina_;
	ov::InferRequest infer_request_;

private:

	Process* process = new Process();
	int input_width = 400;
	int input_height = 400;
	int thread_num = 1;
	std::vector<std::vector<int>> anchors;
};

#endif