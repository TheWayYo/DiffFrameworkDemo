#pragma once
#ifndef __PROCESS_H_
#define __PROCESS_H_

#include <opencv2/opencv.hpp>
#include <iostream>

typedef struct FaceBbox {
	std::vector<int> location;
	std::vector<float> land;
	float score;
	std::vector<float> cov;
} FaceBbox;


class Process
{
public:
	Process() = default;

	~Process() = default;

	int PostProcess();

	cv::Mat PreProcess(const cv::Mat& src_image);

	float iou(const FaceBbox& a, const FaceBbox& b);

	int nms_cpu(std::vector<FaceBbox>& bbox, const float& nms_th, std::vector<FaceBbox>& res);

	void GenetatorAnchors(const int& width, const int& height, std::vector<std::vector<int> >& anchors);

	int BboxPred(const float* bbox, const float* cls, const float* lm, const int& num_anchors, const std::vector<std::vector<int> >& anchors, 
		std::vector<FaceBbox>& bboxes, const int& input_w, const int& input_h, const float& score);

	int PostProcess(std::vector<std::vector<int>> anchors, float* bbox, float* cls, const float* lm, int input_w, int input_h, int pad_img_w, int pad_img_h, int img_w, int img_h, std::vector<FaceBbox>& faceBoxes, float score);


};
#endif
