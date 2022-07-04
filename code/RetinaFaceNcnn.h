#pragma once
#ifndef __RETINA_FACE_NCNN_H_
#define __RETINA_FACE_NCNN_H_
#include "Process.h"
#include <queue>
#include <memory>
#include <string>
#include "ncnn/net.h"


class RetinaFaceNcnn
{
public:
	RetinaFaceNcnn(const std::string& model_path);

	~RetinaFaceNcnn();

	void Init(int thread_num);

	void DetectFace(const cv::Mat& image, std::vector<FaceBbox>& faceboxes);
	
private:
	
	ncnn::Net retina_;
	Process* process = new Process();
	int input_width = 400;
	int input_height = 400;
	int thread_num = 1;
	std::vector<std::vector<int>> anchors;
};

#endif // !1



