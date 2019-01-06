//
// Created by jfajkowski on 01.12.18.
//

#ifndef TABLEWARE_RECOGNITION_UTIL_H
#define TABLEWARE_RECOGNITION_UTIL_H

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

using namespace cv;

double truncate(double value);

void hishow(const String &winname, Mat &I, int hist_size = 256, float lo = 0, float hi = 256, bool uniform = true,
            bool accumulate = false);

std::vector<Point> generatePoints(Mat &I, size_t size, bool grid);

#endif //TABLEWARE_RECOGNITION_UTIL_H
