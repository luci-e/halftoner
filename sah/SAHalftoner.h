#pragma once

#include <opencv/cv.hpp>	
#include <opencv/highgui.h>
#include <string>

class SAHer {
public:
    SAHer(cv::Mat &im8uc1, bool use_external_init = false, float wg = 0.98);
    ~SAHer() {}

    void ComputeSAH(const cv::Mat &sal = cv::Mat(), bool save_intermediates = true);
    cv::Mat GetResult();

private:
    static const int IMG_TYPE;
    cv::Mat src_image_, halftone_image_;
    int w_, h_;
    bool use_external_init_;
    float wg_, wt_;
    void HalfToneInit();
    void HalftoneRead();
    float Objective(const cv::Rect &roi);
};
