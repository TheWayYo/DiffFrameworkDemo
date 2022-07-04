#include "Process.h"


cv::Mat Process::PreProcess(const cv::Mat& src_image) {
	cv::Mat dst_img;
	if (src_image.cols > src_image.rows) {
		dst_img = cv::Mat::zeros(src_image.cols, src_image.cols, CV_8UC3);
	}
	else {
		dst_img = cv::Mat::zeros(src_image.rows, src_image.rows, CV_8UC3);
	}

	dst_img.setTo(cv::Scalar(0, 0, 0));
	cv::Rect roi_rect = cv::Rect(0, 0, src_image.cols, src_image.rows);
	src_image.copyTo(dst_img(roi_rect));
	return dst_img;
}

int Process::PostProcess(std::vector<std::vector<int>> anchors, float* bbox, float* cls, const float* lm, int input_w, int input_h, int pad_img_w, int pad_img_h, int img_w, int img_h, std::vector<FaceBbox>& faceBoxes, float score)
{
	int num_anchors = static_cast<int>(anchors.size());

	std::vector<FaceBbox> bbox1_;
	std::vector<FaceBbox> res_;
	int rst = BboxPred(bbox, cls, lm, num_anchors, anchors, bbox1_, input_w, input_h, score);
	if (bbox1_.size() == 0) {
		return 0;  //
	}
	rst = nms_cpu(bbox1_, 0.4, res_);
	if (res_.size() > 0) {
		float scale_x = (float)pad_img_w / (float)input_w;
		float scale_y = (float)pad_img_h / (float)input_h;
		for (auto& facebox : res_) {
			int raw_ltx = facebox.location[0] * scale_x;
			int raw_lty = facebox.location[1] * scale_y;
			int raw_rbx = facebox.location[2] * scale_x;
			int raw_rby = facebox.location[3] * scale_y;

			int ltx = static_cast<int>(raw_ltx - 0.0 * (raw_rbx - raw_ltx));
			int lty = static_cast<int>(raw_lty - 0.0 * (raw_rby - raw_lty));
			int rbx = static_cast<int>(raw_rbx + 0.0 * (raw_rbx - raw_ltx));
			int rby = static_cast<int>(raw_rby + 0.0 * (raw_rby - raw_lty));

			facebox.location[0] = ltx >= 0 ? ltx : 0;
			facebox.location[1] = lty >= 0 ? lty : 0;
			facebox.location[2] = rbx < img_w ? rbx : img_w - 1;
			facebox.location[3] = rby < img_h ? rby : img_h - 1;

			int face_w = facebox.location[2] - facebox.location[0];
			int face_h = facebox.location[3] - facebox.location[1];

			for (int i = 0; i < 5; i++) {
				facebox.land[2 * i] = facebox.land[2 * i] * scale_x;
				facebox.land[2 * i + 1] = facebox.land[2 * i + 1] * scale_y;
				facebox.land[2 * i] = facebox.land[2 * i] >= 0 ? facebox.land[2 * i] : 0;
				facebox.land[2 * i + 1] = facebox.land[2 * i + 1] >= 0 ? facebox.land[2 * i + 1] : 0;
			}
			faceBoxes.push_back(facebox);
		}
	}
	return 0;
}


void Process::GenetatorAnchors(const int& width, const int& height, std::vector<std::vector<int> >& anchors) {
	int steps[3] = { 8,16,32 };
	int min_sizes[3][2] = { {16,32},{64,128},{256,512} };
	int feature_maps[3][2] = { 0 };
	for (int i = 0; i < 3; i++) {
		feature_maps[i][0] = ceil(height * 1.0 / steps[i]);
		feature_maps[i][1] = ceil(width * 1.0 / steps[i]);
	}
	anchors.clear();
	for (int i = 0; i < 3; i++) {
		int* min_size = min_sizes[i];
		for (int id_y = 0; id_y < feature_maps[i][0]; id_y++) {
			for (int id_x = 0; id_x < feature_maps[i][1]; id_x++)
				for (int k = 0; k < 2; k++) {
					int s_kx = int(min_size[k] * 1.0);
					int s_ky = int(min_size[k] * 1.0);
					int dense_cx = int((id_x + 0.5) * steps[i]);
					int dense_cy = int((id_y + 0.5) * steps[i]);
					std::vector<int> a = { dense_cx, dense_cy, s_kx, s_ky };
					anchors.push_back(a);
				}
		}
	}
}


int Process::nms_cpu(std::vector<FaceBbox>& bbox, const float& nms_th, std::vector<FaceBbox>& res) {
	if (bbox.empty()) {
		return 1;
	}
	sort(bbox.begin(), bbox.end(), [](const FaceBbox& a, const FaceBbox& b) {return (a.score > b.score); });
	while (bbox.size() > 0) {
		res.push_back(bbox[0]);
		auto iter = bbox.begin();
		for (int i = 0; i < bbox.size() - 1; i++) {
			int x1 = std::max(bbox[0].location[0], bbox[i + 1].location[0]);
			int y1 = std::max(bbox[0].location[1], bbox[i + 1].location[1]);
			int x2 = std::min(bbox[0].location[2], bbox[i + 1].location[2]);
			int y2 = std::min(bbox[0].location[3], bbox[i + 1].location[3]);
			int w = std::max(0, x2 - x1 + 1);
			int h = std::max(0, y2 - y1 + 1);
			int area_a = (bbox[0].location[2] - bbox[0].location[0] + 1) * (bbox[0].location[3] - bbox[0].location[1] + 1);
			int area_b = (bbox[i + 1].location[2] - bbox[i + 1].location[0] + 1) * (bbox[i + 1].location[3] - bbox[i + 1].location[1] + 1);
			float over_area = w * h;
			float iou = over_area / (area_a + area_b - over_area);
			if (iou > nms_th) {
				bbox.erase(iter + i + 1);
				i--;
			}
		}
		bbox.erase(iter);
	}
	return 0;
}


int Process::BboxPred(const float* bbox, const float* cls, const float* lm, const int& num_anchors,
	const std::vector<std::vector<int> >& anchors, std::vector<FaceBbox>& bboxes, const int& width, const int& height, const float& score)
{
	float variance[2] = { 0.1, 0.2 };
	for (int i = 0; i < num_anchors; ++i) {
		float s = cls[i * 2 + 1];
		if (s > score) {
			float dx = bbox[i * 4 + 0] * variance[0] * anchors[i][2] + anchors[i][0];
			float dy = bbox[i * 4 + 1] * variance[0] * anchors[i][3] + anchors[i][1];
			float dw = static_cast<float>(exp(bbox[i * 4 + 2] * variance[1]) * anchors[i][2]);
			float dh = static_cast<float>(exp(bbox[i * 4 + 3] * variance[1]) * anchors[i][3]);

			std::vector<float> lms;
			for (int j = 0; j < 5; ++j) {
				float x = lm[(i * 5 + j) * 2 + 0] * anchors[i][2] * variance[0] + anchors[i][0];
				float y = lm[(i * 5 + j) * 2 + 1] * anchors[i][3] * variance[0] + anchors[i][1];
				x = x < 0.0f ? 0.0f : (x > static_cast<float>(width) ? static_cast<float>(width) : x);
				y = y < 0.0f ? 0.0f : (y > static_cast<float>(height) ? static_cast<float>(height) : y);
				lms.push_back(x);
				lms.push_back(y);
			}

			int xmin = static_cast<int>(dx - 0.5 * (dw - 1.0)) > 0 ?
				static_cast<int>(dx - 0.5 * (dw - 1.0)) : 0;
			int ymin = static_cast<int>(dy - 0.5 * (dh - 1.0)) > 0 ?
				static_cast<int>(dy - 0.5 * (dh - 1.0)) : 0;
			int xmax = static_cast<int>(dx + 0.5 * (dw - 1.0)) < width ?
				static_cast<int>(dx + 0.5 * (dw - 1.0)) : width;
			int ymax = static_cast<int>(dy + 0.5 * (dh - 1.0)) < height ?
				static_cast<int>(dy + 0.5 * (dh - 1.0)) : height;

			FaceBbox bbox = { { xmin, ymin, xmax, ymax }, lms, s };
			bboxes.push_back(bbox);
		}
	}
}