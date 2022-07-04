#pragma once
#ifndef __RETINA_FACE_NCNN_H_
#define __RETINA_FACE_NCNN_H_
#include "Process.h"
#include <queue>
#include <memory>
#include <string>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>


class RetinaFaceTvm
{
public:
	RetinaFaceTvm(const std::string& model_path);

	~RetinaFaceTvm();

	void Init(int thread_num);

	void DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceboxes);
	
private:

	DLDevice dev{ kDLCPU, 0 };
	tvm::runtime::Module retina_model;
	tvm::runtime::PackedFunc set_input;
	tvm::runtime::PackedFunc get_output;
	tvm::runtime::PackedFunc load_params;
	tvm::runtime::PackedFunc run;
	Process* process = new Process();
	int input_width = 320;
	int input_height = 320;
	std::vector<std::vector<int>> anchors;
};

#endif // !1



