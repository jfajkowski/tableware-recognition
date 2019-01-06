#include <iostream>
#include "processing.h"
#include "feature_extraction.h"
#include "segmentation.h"
#include "util.h"

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    String paths[] = {
            "./data/resized/all.jpg",
            "./data/resized/fork.jpg",
            "./data/resized/knife.jpg",
            "./data/resized/plate.jpg",
            "./data/resized/spoon.jpg",
    };

    for (const String &path: paths) {
        Mat raw_image = imread(path, 1);
        namedWindow("Raw Image", WINDOW_NORMAL);
        imshow("Raw Image", raw_image);

        namedWindow("Image Histogram", WINDOW_NORMAL);
        hishow("Image Histogram", raw_image);

        Mat gray_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
        toGrayscale(raw_image, gray_image);
        namedWindow("Grayscale Image", WINDOW_NORMAL);
        imshow("Grayscale Image", gray_image);

        Mat lsh[3];
        split(raw_image, lsh);
        for (Mat &channel: lsh) {
            Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
//        threshold(gray_image, binary_image, 128, 255);
            std::vector<Point> points = generatePoints(raw_image, 7, true);
            floodFill(channel, binary_image, points, 16);
//        Mat K = getStructuringElement(MORPH_RECT, Size(3, 3));
//        dilate(binary_image, binary_image, K);
            namedWindow("Binary Image", WINDOW_NORMAL);
            imshow("Binary Image", binary_image);
            waitKey(0);
        }

        Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
//        threshold(gray_image, binary_image, 128, 255);
        std::vector<Point> points = generatePoints(raw_image, 7, true);
        floodFill(gray_image, binary_image, points, 16);
//        Mat K = getStructuringElement(MORPH_RECT, Size(3, 3));
//        dilate(binary_image, binary_image, K);
        namedWindow("Binary Image", WINDOW_NORMAL);
        imshow("Binary Image", binary_image);
        waitKey(0);
    }
    return 0;
}