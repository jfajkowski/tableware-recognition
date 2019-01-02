//
// Created by jfajkowski on 04.12.18.
//

#ifndef TABLEWARE_RECOGNITION_SEGMENTATION_H
#define TABLEWARE_RECOGNITION_SEGMENTATION_H

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

Mat &threshold(Mat &I, Mat &O, uchar low, uchar high);

Mat &erode(Mat &I, Mat &K);

Mat &dilate(Mat &I, Mat &K);

#endif //TABLEWARE_RECOGNITION_SEGMENTATION_H
