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
            O.at<uchar>(row, col) = static_cast<uchar>(pixel >= low && pixel <= high ? SHAPE : BACKGROUND);

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
    if (old_color >= low && old_color <= high) {
        O.at<uchar>(row, col) = new_color;
        floodFill(I, O, row + 1, col, low, high, new_color);
        floodFill(I, O, row, col + 1, low, high, new_color);
        floodFill(I, O, row - 1, col, low, high, new_color);
        floodFill(I, O, row, col - 1, low, high, new_color);
    }
}

Mat &floodFill(Mat &I, Mat &O, const std::vector<Point> &points, uchar level, bool exact = false) {
    CV_Assert(I.type() == CV_8UC1 && O.type() == CV_8UC1 && points.size() <= 255);
    for (size_t i = 0; i < points.size(); ++i) {
        int row = points.at(i).y;
        int col = points.at(i).x;
        if (exact) {
            floodFill(I, O, row, col, level, level, static_cast<uchar>((i + 1) * 255 / points.size()));
        } else {
            uchar low = static_cast<uchar>(truncate(I.at<uchar>(row, col) - level));
            uchar high = static_cast<uchar>(truncate(I.at<uchar>(row, col) + level));
            floodFill(I, O, row, col, low, high, static_cast<uchar>((i + 1) * 255 / points.size()));
        }
    }
    return O;
}

//std::vector<std::unique_ptr<Mat>> split(Mat &I) {
//    CV_Assert(I.type() == CV_16UC1);
//    std::map<ushort, Mat> value_to_matrix;
//    for (int row = 0; row < I.rows; ++row) {
//        for (int col = 0; col < I.cols; ++col) {
//            ushort pixel = I.at<ushort>(row, col);
//            if (value_to_matrix.count(pixel) == 0) {
//                value_to_matrix[pixel] = Mat::zeros(I.size(), CV_8UC1);
//            }
//            value_to_matrix[pixel].at<ushort>(row, col) = SHAPE;
//            bimshow("Dupa", value_to_matrix[pixel]);
//            waitKey(0);
//        }
//    }
//
//    std::vector<std::unique_ptr<Mat>> result;
//    for (auto &pair: value_to_matrix) {
//        bimshow("Dupa", pair.second);
//        waitKey(0);
////        result.push_back(pair.second);
//    }
//    return result;
//}

Mat &erode(Mat &I, Mat &K) {
    CV_Assert(I.type() == CV_8UC1 && K.rows % 2 == 1 && K.cols % 2 == 1);
    Mat O;
    int pad_y = K.rows / 2;
    int pad_x = K.cols / 2;
    copyMakeBorder(I, O, pad_y, pad_y, pad_x, pad_x, BORDER_REPLICATE);
    for (int O_y = pad_y; O_y < O.rows - pad_y; ++O_y) {
        for (int O_x = pad_x; O_x < O.cols - pad_x; ++O_x) {
            int I_y = O_y - pad_y;
            int I_x = O_x - pad_x;
            uchar min = 255;
            for (int K_y = 0; K_y < K.rows; ++K_y) {
                for (int K_x = 0; K_x < K.cols; ++K_x) {
                    if (K.at<uchar>(K_y, K_x) == 1) {
                        uchar pixel = O.at<uchar>(I_y + K_y, I_x + K_x);
                        min = pixel < min ? pixel : min;
                    }
                }
            }
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
            int I_y = O_y - pad_y;
            int I_x = O_x - pad_x;
            uchar max = 0;
            for (int K_y = 0; K_y < K.rows; ++K_y) {
                for (int K_x = 0; K_x < K.cols; ++K_x) {
                    if (K.at<uchar>(K_y, K_x) == 1) {
                        uchar pixel = O.at<uchar>(I_y + K_y, I_x + K_x);
                        max = pixel > max ? pixel : max;
                    }
                }
            }
            I.at<uchar>(I_y, I_x) = max;
        }
    }
    return I;
}
