//
// Created by jfajkowski on 01.12.18.
//

#include "util.h"
#include <vector>

double truncate(double value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return value;
}

void bimshow(const String &winname, Mat &I) {
    CV_Assert(I.type() == CV_8UC1);
    Mat S(I.size(), CV_8UC1);
    for (int row = 0; row < I.rows; ++row) {
        for (int col = 0; col < I.cols; ++col) {
            S.at<uchar>(row, col) = static_cast<uchar>(I.at<uchar>(row, col) == 1 ? 255 : 0);
        }
    }
    imshow(winname, I);
}

void hishow(const String &winname, Mat &I, int hist_size, float lo, float hi, bool uniform, bool accumulate) {
    CV_Assert(I.type() == CV_8UC3);
    std::vector<Mat> bgr_planes;
    split(I, bgr_planes);

    float range[] = {lo, hi};
    const float *hist_range = {range};

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &hist_size, &hist_range, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &hist_size, &hist_range, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &hist_size, &hist_range, uniform, accumulate);

    /// Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / hist_size);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /// Draw for each channel
    for (int i = 1; i < hist_size; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
             Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
             Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }

    /// Display
    imshow(winname, histImage);
}
