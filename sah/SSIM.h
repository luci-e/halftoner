#pragma once

#include <opencv/cv.hpp>	
#include <opencv2/core/types_c.h>

// will process float point input image
IplImage *ssim(IplImage *input1, IplImage *input2);

cv::Mat ssim(cv::Mat &input1, cv::Mat &input2);
