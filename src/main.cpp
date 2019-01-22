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

std::vector<double> image_to_vector(const Mat &I, bool show) {
    Mat focused_image = boundingBox(I);

    if (show) {
        bimshow("Segment", focused_image);
        waitKey(0);
    }

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

std::vector<double> file_to_vector(const std::string &file_path) {
    Mat raw_image = cv::imread(file_path);

    Mat gray_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    toGrayscale(raw_image, gray_image);

    Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    threshold(gray_image, binary_image, 128, 255);

    Mat morph_K = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(binary_image, morph_K);
    erode(binary_image, morph_K);

    return image_to_vector(binary_image, false);
}

void plateDetector(Mat raw_image, Classifier clf) {
    Mat low_pass_K = Mat::ones(Size(5, 5), CV_64F) / 25;
    filter(raw_image, low_pass_K);

    Mat gray_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    toGrayscale(raw_image, gray_image);

    Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    threshold(gray_image, binary_image, 192, 255);

    Mat morph_K = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
    dilate(binary_image, morph_K);
    erode(binary_image, morph_K);

    Mat segmented_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    std::vector<Point> points = generatePoints(segmented_image, 255, false);
    floodFill(binary_image, segmented_image, points, 1, true);

    std::set<uchar> distinct_values;
    for (int r = 0; r < segmented_image.rows; ++r) {
        for (int c = 0; c < segmented_image.cols; ++c) {
            distinct_values.insert(segmented_image.at<uchar>(r, c));
        }
    }
    distinct_values.erase(0);

    for (uchar value: distinct_values) {
        Mat shape = cv::Mat::zeros(raw_image.size(), CV_8UC1);
        for (int r = 0; r < segmented_image.rows; ++r) {
            for (int c = 0; c < segmented_image.cols; ++c) {
                shape.at<uchar>(r, c) = segmented_image.at<uchar>(r, c) == value ? SHAPE : BACKGROUND;
            }
        }

        if (calculateArea(shape) > 5000) {
            Matrix X;
            X.addRow(image_to_vector(shape, true));
            std::cout << clf.predict_proba(X) << std::endl;
        }
    }
}

void cutleryDetector(Mat raw_image, Classifier clf) {
    adjustBrightness(raw_image, -50);
    adjustContrast(raw_image, 50);

    Mat low_pass_K = Mat::ones(Size(5, 5), CV_64F) / 25;
    filter(raw_image, low_pass_K);

    Mat hsl_image = cv::Mat::zeros(raw_image.size(), CV_8UC3);
    toHSL(raw_image, hsl_image);

    Mat lsh[3];
    split(hsl_image, lsh);

    Mat l = lsh[0];
    Mat l_binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    threshold(l, l_binary_image, 16, 255);

    Mat s = lsh[1];
    Mat s_binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    threshold(s, s_binary_image, 0, 63);

    Mat binary_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    bitwise_and(l_binary_image, s_binary_image, binary_image);

    Mat segmented_image = cv::Mat::zeros(raw_image.size(), CV_8UC1);
    std::vector<Point> points = generatePoints(segmented_image, 255, false);
    floodFill(binary_image, segmented_image, points, 1, true);

    std::set<uchar> distinct_values;
    for (int r = 0; r < segmented_image.rows; ++r) {
        for (int c = 0; c < segmented_image.cols; ++c) {
            distinct_values.insert(segmented_image.at<uchar>(r, c));
        }
    }
    distinct_values.erase(0);

    for (uchar value: distinct_values) {
        Mat shape = cv::Mat::zeros(raw_image.size(), CV_8UC1);
        for (int r = 0; r < segmented_image.rows; ++r) {
            for (int c = 0; c < segmented_image.cols; ++c) {
                shape.at<uchar>(r, c) = segmented_image.at<uchar>(r, c) == value ? SHAPE : BACKGROUND;
            }
        }

        Mat morph_small_K = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        dilate(shape, morph_small_K);
        erode(shape, morph_small_K);

        if (calculateArea(shape) < 5000 && calculatePerimeter(shape) > 500) {
            Mat morph_big_K = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
            dilate(shape, morph_big_K);
            erode(shape, morph_big_K);

            Matrix X;
            X.addRow(image_to_vector(shape, true));
            std::cout << clf.predict_proba(X) << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    // Load reference data
    Matrix X;
    Matrix y;
    std::map<std::string, std::vector<double>> name_to_class = {
            {"plate", {1.0, 0.0, 0.0, 0.0}},
            {"fork", {0.0, 1.0, 0.0, 0.0}},
            {"knife", {0.0, 0.0, 1.0, 0.0}},
            {"spoon", {0.0, 0.0, 0.0, 1.0}}
    };

    std::cout << "Loading references for: " << std::endl;
    fs::path targetDir("./data/resized/reference");
    fs::directory_iterator dir_it(targetDir);
    for (fs::path const &dir_path: dir_it) {
        fs::directory_iterator file_it(dir_path);
        std::cout << "\t* " << dir_path.filename().string() << std::endl;
        for (fs::path const &file_path: file_it) {
            X.addRow(file_to_vector(file_path.string()));
            y.addRow(name_to_class[dir_path.filename().string()]);
        }
    }

    std::cout << "Preparing classifier..." << std::endl;
    Classifier clf;
    clf.fit(X, y);
    std::cout << "Classifier accuracy on reference pictures: " << clf.accuracy(X, y) << std::endl;

    String paths[] = {
            "./data/resized/target/all.jpg",
            "./data/resized/target/fork.jpg",
            "./data/resized/target/knife.jpg",
            "./data/resized/target/plate.jpg",
            "./data/resized/target/spoon.jpg",
    };

    for (const String &path: paths) {
        Mat raw_image = cv::imread(path);
        Matrix vectors;

        namedWindow("Raw Image", WINDOW_NORMAL);
        imshow("Raw Image", raw_image);
        waitKey(0);

        plateDetector(raw_image.clone(), clf);
        cutleryDetector(raw_image.clone(), clf);
    }
    return 0;
}