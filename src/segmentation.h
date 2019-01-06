//
// Created by jfajkowski on 04.12.18.
//

#ifndef TABLEWARE_RECOGNITION_SEGMENTATION_H
#define TABLEWARE_RECOGNITION_SEGMENTATION_H

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

using namespace cv;

Mat &threshold(Mat &I, Mat &O, uchar low, uchar high);

Mat &floodFill(Mat &I, Mat &O, const std::vector<Point> &points, uchar level);

Mat &erode(Mat &I, Mat &K);

Mat &dilate(Mat &I, Mat &K);

#endif //TABLEWARE_RECOGNITION_SEGMENTATION_H
