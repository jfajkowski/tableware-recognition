#include <iostream>
#include <map>
#include "processing.h"
#include "feature_extraction.h"
#include "segmentation.h"
#include "util.h"

#include <opencv4/opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

std::vector<double> file_to_vector(const fs::path &file_path) {
    Mat raw_image = cv::imread(file_path.string());
    Mat gray_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    toGrayscale(raw_image, gray_image);
    Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    threshold(gray_image, binary_image, 128, 255);

    double area = calculateArea(binary_image);
    double perimeter = calculatePerimeter(binary_image);
    return {
        calculateShapeCoefficient(area, perimeter),
        calculateMomentInvariant(binary_image, 3),
        calculateMomentInvariant(binary_image, 7)
    };
}

int main(int argc, char **argv) {

    // Load reference data
    Matrix X;
    Matrix y;

    std::map<std::string, std::vector<double>> name_to_class = {
            {"fork", {1.0, 0.0, 0.0, 0.0}},
            {"knife", {0.0, 1.0, 0.0, 0.0}},
            {"plate", {0.0, 0.0, 1.0, 0.0}},
            {"spoon", {0.0, 0.0, 0.0, 1.0}}
    };

    fs::path targetDir("./data/resized/reference");
    fs::directory_iterator dir_it(targetDir);
    for (fs::path const &dir_path: dir_it) {
        fs::directory_iterator file_it(dir_path);
        for (fs::path const &file_path: file_it) {
            X.addRow(file_to_vector(file_path));
            y.addRow(name_to_class[dir_path.filename().string()]);
        }
    }

    std::cout << X << std::endl;
    std::cout << y << std::endl;

//    String paths[] = {
//            "./data/resized/all.jpg",
//            "./data/resized/fork.jpg",
//            "./data/resized/knife.jpg",
//            "./data/resized/plate.jpg",
//            "./data/resized/spoon.jpg",
//    };
//
//    for (const String &path: paths) {
//        Mat raw_image = imread(path);
//        namedWindow("Raw Image", WINDOW_NORMAL);
//        imshow("Raw Image", raw_image);
//
//        namedWindow("Image Histogram", WINDOW_NORMAL);
//        hishow("Image Histogram", raw_image);
//
//        Mat gray_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
//        toGrayscale(raw_image, gray_image);
//        namedWindow("Grayscale Image", WINDOW_NORMAL);
//        imshow("Grayscale Image", gray_image);
//
//        Mat lsh[3];
//        split(raw_image, lsh);
//        for (Mat &channel: lsh) {
//            Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
////        threshold(gray_image, binary_image, 128, 255);
//            std::vector<Point> points = generatePoints(raw_image, 7, true);
//            floodFill(channel, binary_image, points, 16);
////        Mat K = getStructuringElement(MORPH_RECT, Size(3, 3));
////        dilate(binary_image, binary_image, K);
//            namedWindow("Binary Image", WINDOW_NORMAL);
//            imshow("Binary Image", binary_image);
//            waitKey(0);
//        }
//
//        Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
////        threshold(gray_image, binary_image, 128, 255);
//        std::vector<Point> points = generatePoints(raw_image, 7, true);
//        floodFill(gray_image, binary_image, points, 16);
////        Mat K = getStructuringElement(MORPH_RECT, Size(3, 3));
////        dilate(binary_image, binary_image, K);
//        namedWindow("Binary Image", WINDOW_NORMAL);
//        imshow("Binary Image", binary_image);
//        waitKey(0);
//    }
    return 0;
}