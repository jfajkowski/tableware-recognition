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
//            "./data/selected/all.jpg",
            "./data/selected/fork.jpg",
//            "./data/selected/knife.jpg",
            "./data/selected/plate.jpg",
//            "./data/selected/spoon.jpg",
    };
    
    for (const String &path: paths) {
        Mat raw_image = imread(path, 1);
        namedWindow("Raw Image", WINDOW_NORMAL);
        imshow("Raw Image", raw_image);

        namedWindow("Image Histogram", WINDOW_NORMAL);
        hishow("Image Histogram", raw_image);

        Mat gray_image(raw_image.size(), CV_8UC1);
        toGrayscale(raw_image, gray_image);
        namedWindow("Grayscale Image", WINDOW_NORMAL);
        imshow("Grayscale Image", gray_image);

        Mat binary_image(raw_image.size(), CV_8UC1);
        threshold(gray_image, binary_image, 200, 255);
        Mat K = getStructuringElement(MORPH_RECT, Size(3, 3));
        erode(binary_image, binary_image, K);
        namedWindow("Binary Image", WINDOW_NORMAL);
        bimshow("Binary Image", binary_image);

        waitKey(0);
    }
    return 0;
}