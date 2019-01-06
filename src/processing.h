//
// Created by jfajkowski on 01.12.18.
//

#ifndef TABLEWARE_RECOGNITION_ADJUSTMENT_H
#define TABLEWARE_RECOGNITION_ADJUSTMENT_H

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

Mat &toGrayscale(const Mat &I, Mat &O);

Mat &toHSL(const Mat &I, Mat &O);

Mat &adjustBrightness(Mat &I, int value);

Mat &adjustContrast(Mat &I, int value);

Mat &filter(Mat &I, Mat &K);

#endif //TABLEWARE_RECOGNITION_ADJUSTMENT_H
