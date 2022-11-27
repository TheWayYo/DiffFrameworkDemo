#include <iostream>
#include "opencv2/opencv.hpp"
#include <chrono>
//#include "RetinaFaceVino.h"
//#include "RetinaFaceNcnn.h"
#include "RetinaFaceTvm.h"

int main() {
	RetinaFaceTvm* retinaFace = new RetinaFaceTvm("");
	int count = 10;
	cv::Mat image = cv::imread("");
	if (image.channels() == 4) {
		cv::cvtColor(image, image, CV_RGBA2BGR);
	}
	auto total_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < count; i++) {
		std::vector<FaceBbox> faceboxes;
		retinaFace->DetectFace(image, faceboxes);
		
		for (auto& box : faceboxes) {
			cv::rectangle(image, cv::Point((int)box.location[0], (int)box.location[1]), cv::Point((int)box.location[2], (int)box.location[3]), (0, 255, 0), 2, 8, 0);
			for (int j = 0; j < 5; j++) {
				int point_x = box.land[2 * j];
				int point_y = box.land[2 * j + 1];
				cv::circle(image, cv::Point(point_x, point_y), 2, cv::Scalar(0, 255, 255), -1);
			}
		}
	}
	auto total_end = std::chrono::high_resolution_clock::now();
	auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
	std::cout << "total time cost = " << total_duration / (count * 1000) << " ms" << std::endl;
	cv::imshow("test", image);
	int q = cv::waitKey(0);
	return 0;
}