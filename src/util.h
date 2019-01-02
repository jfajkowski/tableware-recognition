//
// Created by jfajkowski on 01.12.18.
//

#ifndef TABLEWARE_RECOGNITION_UTIL_H
#define TABLEWARE_RECOGNITION_UTIL_H

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

double truncate(double value);

void bimshow(const String& winname, Mat &I);

void hishow(const String &winname, Mat &I, int hist_size = 256, float lo = 0, float hi = 256, bool uniform = true,
            bool accumulate = false);

#endif //TABLEWARE_RECOGNITION_UTIL_H
