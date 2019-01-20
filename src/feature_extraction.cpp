//
// Created by jfajkowski on 04.12.18.
//

#include "feature_extraction.h"
#include "processing.h"
#include "util.h"

double calculateArea(const Mat &I) {
    CV_Assert(I.type() == CV_8UC1);
    double area = 0;
    for (int y = 0; y < I.rows; ++y) {
        for (int x = 0; x < I.cols; ++x) {
            bool is_area = I.at<uchar>(y, x) == 1;
            if (is_area) {
                ++area;
            }
        }
    }
    return area;
}

double calculatePerimeter(const Mat &I) {
    CV_Assert(I.type() == CV_8UC1);
    double perimeter = 0;
    for (int y = 1; y < I.rows - 1; ++y) {
        for (int x = 1; x < I.cols - 1; ++x) {
            bool is_perimeter = I.at<uchar>(y, x) == 1
                                && (I.at<uchar>(y - 1, x - 1) != I.at<uchar>(y + 1, x + 1)
                                    || I.at<uchar>(y - 1, x) != I.at<uchar>(y + 1, x)
                                    || I.at<uchar>(y - 1, x + 1) != I.at<uchar>(y + 1, x - 1)
                                    || I.at<uchar>(y, x - 1) != I.at<uchar>(y, x + 1));
            if (is_perimeter) {
                ++perimeter;
            }
        }
    }
    return perimeter;
}

double calculateShapeCoefficient(double area, double perimeter) {
    return (perimeter / (2 * sqrt(M_PI * area))) - 1;
}

double m(const Mat &I, double p, double q) {
    CV_Assert(I.type() == CV_8UC1);
    double m = 0;
    for (int j = 0; j < I.rows; ++j) {
        for (int i = 0; i < I.cols; ++i) {
            m += pow((double) i, p) * pow((double) j, q) * (double) I.at<uchar>(j, i);
        }
    }
    return m;
}

double M(const Mat &I, double p, double q) {
    CV_Assert(I.type() == CV_8UC1);
    double i_ = m(I, 1, 0) / m(I, 0, 0);
    double j_ = m(I, 0, 1) / m(I, 0, 0);
    double m = 0;
    for (int j = 0; j < I.rows; ++j) {
        for (int i = 0; i < I.cols; ++i) {
            m += pow((double) i - i_, p) * pow((double) j - j_, q) * (double) I.at<uchar>(j, i);
        }
    }
    return m;
}

double calculateMomentInvariant(const Mat &I, int n) {
    CV_Assert(I.type() == CV_8UC1);
    switch (n) {
        case 3:
            return (pow(M(I, 3, 0) - 3 * M(I, 1, 2), 2) + pow(3 * M(I, 2, 1) - M(I, 0, 3), 2)) / pow(M(I, 0, 0), 5);
        case 7:
            return (M(I, 2, 0) * M(I, 0, 2) - pow(M(I, 1, 1), 2)) / pow(M(I, 0, 0), 4);
        default:
            return 0;
    }
}

double calculateTilt(const Mat &I) {
    CV_Assert(I.type() == CV_8UC1);
    int n = I.rows, s = 0, w = I.cols, e = 0;
    for (int y = 1; y < I.rows; ++y) {
        for (int x = 1; x < I.cols; ++x) {
            n = I.at<uchar>(y - 1, x) == I.at<uchar>(y, x) ? n : min(n, y);
            s = I.at<uchar>(y - 1, x) == I.at<uchar>(y, x) ? s : max(s, y);
            w = I.at<uchar>(y, x - 1) == I.at<uchar>(y, x) ? w : min(w, x);
            e = I.at<uchar>(y, x - 1) == I.at<uchar>(y, x) ? e : max(e, x);
        }
    }

    Mat O(I, Rect(w, n, e - w, s - n));

    double rect_mid_y = (double) (s - n) / 2, rect_mid_x = (double) (e - w) / 2;
    double obj_mid_y = m(O, 0, 1) / m(O, 0, 0), obj_mid_x = m(O, 1, 0) / m(O, 0, 0);

    double a = rect_mid_y - obj_mid_y;
    double b = obj_mid_x - rect_mid_x;

    double alpha = atan(a / b) * (180.0 / M_PI);

    if (b <= 0) {
        return 180.0 + alpha;
    } else if (a < 0) {
        return 360.0 + alpha;
    } else {
        return alpha;
    }
}

Mat boundingBox(const Mat &I) {
    CV_Assert(I.type() == CV_8UC1);
    int n = I.rows, s = 0, w = I.cols, e = 0;
    for (int y = 1; y < I.rows; ++y) {
        for (int x = 1; x < I.cols; ++x) {
            n = I.at<uchar>(y - 1, x) == I.at<uchar>(y, x) ? n : min(n, y);
            s = I.at<uchar>(y - 1, x) == I.at<uchar>(y, x) ? s : max(s, y);
            w = I.at<uchar>(y, x - 1) == I.at<uchar>(y, x) ? w : min(w, x);
            e = I.at<uchar>(y, x - 1) == I.at<uchar>(y, x) ? e : max(e, x);
        }
    }
    return Mat(I, Rect(w, n, e - w, s - n));
}
