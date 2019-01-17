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

std::vector<Point> generatePoints(Mat &I, size_t size, bool grid) {
    RNG rng(0xFFFFFFFF);
    std::vector<Point> points;
    if (grid) {
        int grid_x = static_cast<int>(I.cols / size);
        int grid_y = static_cast<int>(I.rows / size);
        for (int row = 0; row < I.rows; row += grid_y) {
            for (int col = 0; col < I.cols; col += grid_x) {
                points.emplace_back(rng.uniform(row, row + grid_y), rng.uniform(col, col + grid_x));
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            points.emplace_back(rng.uniform(0, I.rows), rng.uniform(0, I.rows));
        }
    }
    return points;
}

Matrix::Matrix()
        : Matrix(0, 0) {

}

Matrix::Matrix(size_t rows, size_t cols)
        : mRows(rows),
          mCols(cols),
          mData(rows * cols) {
}

Matrix::Matrix(const Matrix &matrix) {
    mRows = matrix.mRows;
    mCols = matrix.mCols;
    mData = matrix.mData;
}

double &Matrix::operator()(size_t i, size_t j) {
    return mData[i * mCols + j];
}

double Matrix::operator()(size_t i, size_t j) const {
    return mData[i * mCols + j];
}

void Matrix::addRow(std::vector<double> row) {
    if (mRows == 0) {
        mCols = row.size();
    } else if (row.size() != mCols) {
        throw std::invalid_argument("Row should have same column number as matrix!");
    }
    mData.reserve(mData.size() + row.size());
    mData.insert(mData.end(), row.begin(), row.end());
    mRows += 1;
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    for (size_t i = 0; i < matrix.mRows; ++i) {
        for (size_t j = 0; j < matrix.mCols; ++j) {
            os << matrix(i, j);
            if (j != matrix.mCols - 1) {
                os << ",";
            }
        }
        os << std::endl;
    }
    os << std::endl;
    return os;
}
