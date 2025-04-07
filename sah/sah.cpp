#include <opencv/cv.hpp>	
#include <opencv/highgui.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
#include "SSIM.h"
#include "Ostromoukhov01.h"
#include "SAHalftoner.h"

using namespace std;

int main(int argc, char** argv)
{
    bool use_external_init = false;
    float wg = 0.98f;

	if(argc<3){
		cout<<"Usage: cvHalftone source_path output_path"<<endl;
		return -1;
	}

    if(argc == 4) {
        use_external_init = bool(atoi(argv[3]));
        std::cout << "Using external init: " << use_external_init << std::endl;
    }else{
        std::cout << "No params, using external init: " << use_external_init << std::endl;
    }

    // check if wg has been passed
    if(argc == 5) {
        wg = float(atof(argv[4]));
        std::cout << "Using wg: " << wg << std::endl;
        if(wg < 0 || wg > 1) {
            std::cerr << "wg must be between 0 and 1" << std::endl;
            return -1;
        }
    }

	cv::Mat im = cv::imread(argv[1], 0);
    // printf("Image read\n");
    SAHer saher(im, use_external_init, wg);
    saher.ComputeSAH();
    cv::Mat resImg = saher.GetResult();
    // cv::imshow("result", resImg);
    resImg.convertTo(resImg, CV_8UC3, 255.0);
    cv::imwrite(argv[2], resImg);
    cv::waitKey();
	return 0;
}