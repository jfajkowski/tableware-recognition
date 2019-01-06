//
// Created by jfajkowski on 04.12.18.
//

#include "segmentation.h"
#include "util.h"

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

void floodFill(Mat &I, Mat &O, int row, int col, uchar low, uchar high, uchar new_color) {
    if (row < 0 || row >= I.rows) {
        return;
    }
    if (col < 0 || col >= I.cols) {
        return;
    }
    if (O.at<uchar>(row, col) != 0) {
        return;
    }

    uchar old_color = I.at<uchar>(row, col);
    if (old_color > low && old_color < high) {
        O.at<uchar>(row, col) = new_color;
        floodFill(I, O, row + 1, col, low, high, new_color);
        floodFill(I, O, row, col + 1, low, high, new_color);
        floodFill(I, O, row - 1, col, low, high, new_color);
        floodFill(I, O, row, col - 1, low, high, new_color);
    }
}

Mat &floodFill(Mat &I, Mat &O, const std::vector<Point> &points, uchar level) {
    CV_Assert(I.type() == CV_8UC1 && points.size() <= 255);
    for (size_t i = 1; i < points.size(); ++i) {
        int row = points.at(i).y;
        int col = points.at(i).x;
        uchar low = static_cast<uchar>(truncate(I.at<uchar>(row, col) - level));
        uchar high = static_cast<uchar>(truncate(I.at<uchar>(row, col) + level));
        floodFill(I, O, row, col, low, high, static_cast<uchar>(i * 255 / points.size()));
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
