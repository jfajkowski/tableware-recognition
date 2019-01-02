//
// Created by jfajkowski on 04.12.18.
//

#include "segmentation.h"

Mat &threshold(Mat &I, Mat &O, uchar low, uchar high) {
    CV_Assert(I.type() == CV_8UC1);
    for (int row = 0; row < I.rows; ++row) {
        for (int col = 0; col < I.cols; ++col) {
            uchar pixel = I.at<uchar>(row, col);
            O.at<uchar>(row, col) = static_cast<uchar>(pixel >= low && pixel <= high ? 255 : 0);

        }
    }
    return O;
}

Mat &erode(Mat &I, Mat &K) {
    CV_Assert(I.type() == CV_8UC1 && K.rows % 2 == 1 && K.cols % 2 == 1);
    Mat O;
    int pad_y = K.rows / 2;
    int pad_x = K.cols / 2;
    copyMakeBorder(I, O, pad_y, pad_y, pad_x, pad_x, BORDER_REPLICATE);
    for (int O_y = pad_y; O_y < O.rows - pad_y; ++O_y) {
        for (int O_x = pad_x; O_x < O.cols - pad_x; ++O_x) {
            uchar min = 255;
            for (int K_y = 0; K_y < K.rows; ++K_y) {
                for (int K_x = 0; K_x < K.cols; ++K_x) {
                    if (K.at<uchar>(K_y, K_x) == 1) {
                        uchar pixel = O.at<uchar>(O_y + K_y, O_x + K_x);
                        min = pixel < min ? pixel : min;
                    }
                }
            }
            int I_y = O_y - pad_y;
            int I_x = O_x - pad_x;
            I.at<uchar>(I_y, I_x) = min;
        }
    }
    return I;
}

Mat &dilate(Mat &I, Mat &K) {
    CV_Assert(I.type() == CV_8UC1 && K.rows % 2 == 1 && K.cols % 2 == 1);
    Mat O;
    int pad_y = K.rows / 2;
    int pad_x = K.cols / 2;
    copyMakeBorder(I, O, pad_y, pad_y, pad_x, pad_x, BORDER_REPLICATE);
    for (int O_y = pad_y; O_y < O.rows - pad_y; ++O_y) {
        for (int O_x = pad_x; O_x < O.cols - pad_x; ++O_x) {
            uchar max = 0;
            for (int K_y = 0; K_y < K.rows; ++K_y) {
                for (int K_x = 0; K_x < K.cols; ++K_x) {
                    if (K.at<uchar>(K_y, K_x) == 1) {
                        uchar pixel = O.at<uchar>(O_y + K_y, O_x + K_x);
                        max = pixel > max ? pixel : max;
                    }
                }
            }
            int I_y = O_y - pad_y;
            int I_x = O_x - pad_x;
            I.at<uchar>(I_y, I_x) = max;
        }
    }
    return I;
}
