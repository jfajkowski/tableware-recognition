//
// Created by jfajkowski on 01.12.18.
//

#include "processing.h"
#include "util.h"

Mat &toGrayscale(const Mat &I, Mat &O) {
    CV_Assert(I.type() == CV_8UC3 && O.type() == CV_8UC1 && I.size() == O.size());
    for (int row = 0; row < I.rows; ++row) {
        for (int col = 0; col < I.cols; ++col) {
            Vec3b pixel = I.at<Vec3b>(row, col);
            O.at<uchar>(row, col) = static_cast<uchar>(((double) pixel[0] + (double) pixel[1] + (double) pixel[2]) / 3);
        }
    }
    return O;
}

void adjustBrightness(Vec3b &pixel, int value) {
    pixel[0] = static_cast<uchar>(truncate(pixel[0] + value));
    pixel[1] = static_cast<uchar>(truncate(pixel[1] + value));
    pixel[2] = static_cast<uchar>(truncate(pixel[2] + value));
}

Mat &adjustBrightness(Mat &I, int value) {
    CV_Assert(I.type() == CV_8UC3);
    I.forEach<Vec3b>(
            [value](Vec3b &pixel, const int *position) -> void {
                adjustBrightness(pixel, value);
            });
    return I;
}

void adjustContrast(Vec3b &pixel, double factor) {
    pixel[0] = static_cast<uchar>(truncate(factor * (pixel[0] - 128) + 128));
    pixel[1] = static_cast<uchar>(truncate(factor * (pixel[1] - 128) + 128));
    pixel[2] = static_cast<uchar>(truncate(factor * (pixel[2] - 128) + 128));
}

Mat &adjustContrast(Mat &I, int value) {
    CV_Assert(I.type() == CV_8UC3);
    double factor = (259 * ((double) value + 255)) / (255 * (259 - (double) value));
    I.forEach<Vec3b>(
            [factor](Vec3b &pixel, const int *position) -> void {
                adjustContrast(pixel, factor);
            });
    return I;
}

Mat &filter(Mat &I, Mat &K) {
    CV_Assert(I.type() == CV_8UC3 && K.rows % 2 == 1 && K.cols % 2 == 1);
    Mat O;
    int pad_y = K.rows / 2;
    int pad_x = K.cols / 2;
    copyMakeBorder(I, O, pad_y, pad_y, pad_x, pad_x, BORDER_REPLICATE);
    for (int c = 0; c < 3; c++) {
        for (int O_y = pad_y; O_y < O.rows - pad_y; ++O_y) {
            for (int O_x = pad_x; O_x < O.cols - pad_x; ++O_x) {
                double accumulator = 0;
                for (int K_y = 0; K_y < K.rows; ++K_y) {
                    for (int K_x = 0; K_x < K.cols; ++K_x) {
                        accumulator += O.at<Vec3b>(O_y + K_y, O_x + K_x)[c] * K.at<double>(K_y, K_x);
                    }
                }
                int I_y = O_y - pad_y;
                int I_x = O_x - pad_x;
                I.at<Vec3b>(I_y, I_x)[c] = static_cast<uchar>(truncate(accumulator));
            }
        }
    }
    return I;
}

