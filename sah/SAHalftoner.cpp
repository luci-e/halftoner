#include "SAHalftoner.h"
#include "Ostromoukhov01.h"
#include "SSIM.h"
#include <vector>
#include <algorithm>
#include "utils.h"
#include <math.h>       /* pow */
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string> // the C++ Standard String Class
#include <sstream>

#include <chrono>
using namespace std::chrono;

const int SAHer::IMG_TYPE = CV_32FC1;

SAHer::SAHer(cv::Mat &im8uc1, bool use_external_init, float wg) {
    w_ = im8uc1.cols;
    h_ = im8uc1.rows;
    use_external_init_ = use_external_init;
    im8uc1.convertTo(this->src_image_, IMG_TYPE, 1./255.);
    wg_ = wg;
    wt_ = 1 - wg_;
}

float SAHer::Objective(const cv::Rect &roi) {
    int blur_range = 11;
    cv::Point lu(std::max(0, roi.x - 2*blur_range), std::max(0, roi.y - 2*blur_range)), rd(std::min(w_, roi.x + roi.width + 2*blur_range), std::min(h_, roi.y + roi.height + 2*blur_range));
    cv::Point offset1(blur_range, blur_range);

    if ( roi.x - blur_range < 0) {
        offset1.x = 0;
    } else if (roi.x - blur_range*2 < 0) {
        offset1.x = roi.x - blur_range;
    }
    if ( roi.y - blur_range < 0) {
        offset1.y = 0;
    } else if (roi.y - blur_range*2 < 0) {
        offset1.y = roi.y - blur_range;
    }
    cv::Rect new_roi(lu, rd);
    cv::Mat src_roi = src_image_(new_roi), halftone_roi = halftone_image_(new_roi);

    cv::Rect sub_roi = cv::Rect(offset1.x, offset1.y, std::min(roi.width + 2*blur_range, new_roi.width - offset1.x), std::min(roi.height + 2*blur_range, new_roi.height - offset1.y));
    //info() << "begin" << roi.x << roi.y << roi.width << roi.height << offset1.x << offset1.y << new_roi.x << new_roi.y << new_roi.width << new_roi.height;
    // info() << "Computing SSIM";
    cv::Mat ssim_map = ssim(src_roi, halftone_roi)(sub_roi);
    // info () << "SSIM computed";
    float mean_ssim = float(cv::mean(ssim_map)[0]);

    cv::Mat gI, gH, se;
    cv::GaussianBlur(src_roi, gI, cv::Size(blur_range, blur_range), 0);
    cv::GaussianBlur(halftone_roi, gH, cv::Size(blur_range, blur_range), 0);
    cv::subtract(gI(sub_roi), gH(sub_roi), se);
    cv::multiply(se, se, se);

    float gaussian_diff = float(cv::mean(se)[0]);

    return wg_*gaussian_diff + wt_*(1.f - mean_ssim);
}

void SAHer::HalfToneInit() {
    halftone_image_ = OstromoukhovHalftone(src_image_);
    halftone_image_.convertTo(halftone_image_, IMG_TYPE, 1.);
    std::cout << "Halftone initialized using Ostromokhouv method" << std::endl;
}

void SAHer::HalftoneRead() {
    halftone_image_ = cv::imread( "./halftoned_custom.png", 0);
    halftone_image_.convertTo(halftone_image_, IMG_TYPE, 1./255.);
    std::cout << "Read halftone image" << std::endl;
}

void SAHer::ComputeSAH(const cv::Mat &sal, bool save_intermediates) {
    if (!use_external_init_){
        HalfToneInit();
    }else{
        HalftoneRead();
    }

    std::srand(std::time(nullptr));
    // printf("HalfToneInit done\n");

     cv::Mat intermediateImage;

    //float e_old = Objective();
    bool use_sal = (sal.cols && sal.rows);
    int block_size = 2;
    float temperature = .2f;
    float AnnealFactor = .8f;

    int n_iter = 0;
    do {
        auto start = high_resolution_clock::now();

        for (int block_i = 0; block_i < h_; block_i += block_size) for (int block_j = 0; block_j < w_; block_j += block_size) {

            std::vector<std::pair<int,int> > b_indices, w_indices;
            for (int ii = 0; ii < block_size && block_i + ii < h_; ii++) {
                for (int jj = 0; jj < block_size && block_j + jj < w_; jj++) {
                    int i = block_i + ii, j = block_j + jj;
                    if (halftone_image_.at<float>(i,j) > 0) w_indices.push_back(std::pair<int,int>(i, j));
                    else b_indices.push_back(std::pair<int,int>(i, j));
                }
            }

            if (b_indices.empty() || w_indices.empty()) continue;
            // else try block_size x block_size times of swap.
            cv::Rect roi(block_j, block_i, std::min(block_size,w_ - block_j ), std::min(block_size, h_ - block_i));
            // printf("Computing Objective\n");
            float e_old = Objective(roi);
            // printf("Objective computed\n");
            

            int exhange_times = use_sal ? round(block_size * block_size * cv::mean(sal(roi))[0]) : block_size * block_size;
            for (int k = 0; k < exhange_times; k++){
                int rand1 = rand() % b_indices.size(), rand2 = rand() % w_indices.size();
                std::pair<int,int> idx1 = b_indices[rand1], idx2 = w_indices[rand2];
                halftone_image_.at<float>(idx1.first, idx1.second) = 1;
                halftone_image_.at<float>(idx2.first, idx2.second) = 0;
                float e_new = Objective(roi);
                float delta_e = e_new - e_old;
                if ( delta_e < 0.f || rand_float() < exp( - delta_e / temperature*w_*h_ ) ) {
                    // accept
                    e_old = e_new;
                    b_indices[rand1] = idx2;
                    w_indices[rand2] = idx1;
                } else {
                    // reject and undo swap
                    halftone_image_.at<float>(idx1.first, idx1.second) = 0;
                    halftone_image_.at<float>(idx2.first, idx2.second) = 1;
                }
            }

            n_iter += exhange_times;

            // std::cout << "Current similarity " << e_old << std::endl;
        }

        std::stringstream ss;

        ss << "./intermediates/" << n_iter << ".png";

        halftone_image_.convertTo(intermediateImage, CV_8UC3, 255.0);
        cv::imwrite(ss.str(), intermediateImage);

        temperature *= AnnealFactor;

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        // std::cout << "Temperature: " << temperature << std::endl;
        // std::cout << duration.count()/pow(10,6) << std::endl;
    } while (temperature > 0.15f);
    // } while (1);


    return;
}

cv::Mat SAHer::GetResult() {
    return halftone_image_;
}
