//
// Created by jfajkowski on 04.12.18.
//

#ifndef TABLEWARE_RECOGNITION_FEATURE_EXTRACTION_H
#define TABLEWARE_RECOGNITION_FEATURE_EXTRACTION_H

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

double calculateArea(const Mat &I);
double calculatePerimeter(const Mat &I);
double calculateShapeCoefficient(double area, double perimeter);
double calculateMomentInvariant(const Mat &I, int n);
double calculateTilt(const Mat &I);

#endif //TABLEWARE_RECOGNITION_FEATURE_EXTRACTION_H
