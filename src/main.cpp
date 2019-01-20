#include <iostream>
#include <map>
#include "processing.h"
#include "feature_extraction.h"
#include "segmentation.h"
#include "util.h"
#include "classifier.h"

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

    Mat K = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(binary_image, K);

    Mat focused_image = boundingBox(binary_image);

//    bimshow("A", focused_image);
//    waitKey(0);

    double area = calculateArea(focused_image);
    double perimeter = calculatePerimeter(focused_image);
    Moments m = moments(focused_image, true);
    return {
        calculateShapeCoefficient(area, perimeter),
//        m.mu20, m.mu11, m.mu02, m.mu30, m.mu21, m.mu12, m.mu03,
//        m.nu20, m.nu11, m.nu02, m.nu30, m.nu21, m.nu12, m.nu03,
        calculateMomentInvariant(focused_image, 3),
        calculateMomentInvariant(focused_image, 7)
    };
}

//for line in input.split('\n'):
//     v = list(map(float, line.split(',')))
//     a.append(v[0])
//     b.append(v[1])

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

    std::cout << "Loading references for: " << std::endl;
    fs::path targetDir("./data/resized/reference");
    fs::directory_iterator dir_it(targetDir);
    for (fs::path const &dir_path: dir_it) {
        fs::directory_iterator file_it(dir_path);
        std::cout << "\t* " << dir_path.filename().string() << std::endl;
        for (fs::path const &file_path: file_it) {
            X.addRow(file_to_vector(file_path));
            y.addRow(name_to_class[dir_path.filename().string()]);
        }
    }

    std::cout << "Preparing classifier..." << std::endl;
    Matrix X_norm;
    for (size_t i = 0; i < X.cols(); ++i) {
        X_norm.addCol(X.getCol(i).normalize());
    }
    Classifier clf;
    clf.fit(X_norm, y);
    std::cout << "Classifier accuracy on reference pictures: " << clf.accuracy(X_norm, y) << std::endl;

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