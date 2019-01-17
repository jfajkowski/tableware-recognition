//
// Created by jfajkowski on 01.12.18.
//

#ifndef TABLEWARE_RECOGNITION_UTIL_H
#define TABLEWARE_RECOGNITION_UTIL_H

#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <ostream>

using namespace cv;

double truncate(double value);

void hishow(const String &winname, Mat &I, int hist_size = 256, float lo = 0, float hi = 256, bool uniform = true,
            bool accumulate = false);

std::vector<Point> generatePoints(Mat &I, size_t size, bool grid);

class Matrix {
private:
    size_t mRows;
    size_t mCols;
    std::vector<double> mData;

public:
    Matrix();

    Matrix(size_t rows, size_t cols);

    Matrix(const Matrix &matrix);

    double &operator()(size_t i, size_t j);

    double operator()(size_t i, size_t j) const;

    void addRow(std::vector<double> row);

    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);
};

#endif //TABLEWARE_RECOGNITION_UTIL_H
