//
// Created by jfajkowski on 04.12.18.
//

#ifndef TABLEWARE_RECOGNITION_SEGMENTATION_H
#define TABLEWARE_RECOGNITION_SEGMENTATION_H

#define BACKGROUND (uchar) 0
#define SHAPE (uchar) 1

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

using namespace cv;

Mat &threshold(Mat &I, Mat &O, uchar low, uchar high);

Mat &floodFill(Mat &I, Mat &O, const std::vector<Point> &points, uchar level, bool exact);

std::vector<std::unique_ptr<Mat>> split(Mat &I);

Mat &erode(Mat &I, Mat &K);

Mat &dilate(Mat &I, Mat &K);

#endif //TABLEWARE_RECOGNITION_SEGMENTATION_H
