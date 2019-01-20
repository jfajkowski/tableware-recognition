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

Vector::Vector(const std::vector<double> &vector)
        : _size(vector.size()),
          _data(vector) {

}

Vector::Vector(size_t size)
        : _size(size),
          _data(size) {

}

std::vector<double> Vector::unwrap() const {
    return std::vector<double>(_data);
}

Vector Vector::normalize() const {
    double sum_mean = 0;
    for (auto v: _data) {
        sum_mean += v;
    }
    double mean = sum_mean / _size;

    double sum_variance = 0;
    for (auto v: _data) {
        sum_variance += std::pow(v - mean, 2);
    }
    double variance = sum_variance / _size;

    auto result = unwrap();
    for (auto &v: result) {
        v = (v - mean) / std::sqrt(variance);
    }
    return Vector(result);
}

size_t Vector::argmax() const {
    double max = 0;
    size_t max_index = 0;
    for (size_t j = 0; j < _size; ++j) {
        if (max < operator[](j)) {
            max = operator[](j);
            max_index = j;
        }
    }
    return max_index;
}

Vector Vector::operator+(const Vector &b) const {
    std::vector<double> result;
    for (size_t i = 0; i < _size; ++i) {
        result.push_back(_data[i] + b._data[i]);
    }
    return Vector(result);
}

Vector &Vector::operator+=(const Vector &b) {
    for (size_t i = 0; i < _size; ++i) {
        _data[i] = _data[i] + b._data[i];
    }
    return *this;
}

Vector Vector::operator/(const double &b) const {
    std::vector<double> result;
    for (size_t i = 0; i < _size; ++i) {
        result.push_back(_data[i] / b);
    }
    return Vector(result);
}

Vector &Vector::operator/=(const double &b) {
    for (size_t i = 0; i < _size; ++i) {
        _data[i] = _data[i] / b;
    }
    return *this;
}

double &Vector::operator[](size_t i) {
    return _data[i];
}

double Vector::operator[](size_t i) const {
    return _data[i];
}

bool Vector::operator==(const Vector &rhs) const {
    return _size == rhs._size &&
           _data == rhs._data;
}

std::ostream &operator<<(std::ostream &os, const Vector &row) {
    for (size_t i = 0; i < row._size; ++i) {
        os << row[i];
        if (i != row._size - 1) {
            os << ",";
        }
    }
    os << std::endl;
    return os;
}

size_t Vector::size() const {
    return _size;
}

Matrix::Matrix()
        : Matrix(0, 0) {

}

Matrix::Matrix(size_t rows, size_t cols)
        : _rows(rows),
          _cols(cols),
          _data(rows * cols) {
}

Matrix::Matrix(const Matrix &matrix) {
    _rows = matrix._rows;
    _cols = matrix._cols;
    _data = matrix._data;
}

double &Matrix::operator()(size_t i, size_t j) {
    return _data[i * _cols + j];
}

double Matrix::operator()(size_t i, size_t j) const {
    return _data[i * _cols + j];
}

Vector Matrix::getRow(size_t n) const {
    if (n >= _rows) {
        throw std::invalid_argument("There are not enough rows in matrix!");
    }
    auto first = _data.begin() + n * _cols;
    auto last = _data.begin() + (n + 1) * _cols;
    return Vector(std::vector<double>(first, last));
}

void Matrix::addRow(Vector row) {
    auto row_data = row.unwrap();
    if (_rows == 0) {
        _cols = row_data.size();
    } else if (row_data.size() != _cols) {
        throw std::invalid_argument("Row should have same column number as matrix!");
    }
    _data.reserve(_data.size() + row_data.size());
    _data.insert(_data.end(), row_data.begin(), row_data.end());
    _rows += 1;
}

int Matrix::rowPosition(Vector row) {
    for (size_t i = 0; i < _rows; ++i) {
        if (getRow(i) == row) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

Vector Matrix::getCol(size_t n) const {
    if (n >= _rows) {
        throw std::invalid_argument("There are not enough cols in matrix!");
    }
    Vector col(_rows);
    for (size_t i = 0; i < _rows; ++i) {
        col[i] = operator()(i, n);
    }
    return col;
}

void Matrix::addCol(Vector col) {
    if (_cols == 0) {
        _rows = col.size();
    } else if (col.size() != _rows) {
        throw std::invalid_argument("Cols should have same row number as matrix!");
    }
    _data.reserve(_data.size() + col.size());
    _cols += 1;
    for (size_t i = 0; i < col.size(); ++i) {
        _data.insert(_data.end() - _cols * i, col[_rows - 1 - i]);
    }
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    for (size_t i = 0; i < matrix._rows; ++i) {
        for (size_t j = 0; j < matrix._cols; ++j) {
            os << matrix(i, j);
            if (j != matrix._cols - 1) {
                os << ",";
            }
        }
        os << std::endl;
    }
    os << std::endl;
    return os;
}

void bimshow(const String &winname, Mat &I) {
    CV_Assert(I.type() == CV_8UC1);
    Mat S(I.size(), CV_8UC1);
    for (int row = 0; row < I.rows; ++row) {
        for (int col = 0; col < I.cols; ++col) {
            S.at<uchar>(row, col) = static_cast<uchar>(I.at<uchar>(row, col) == 1 ? 255 : 0);
        }
    }
    imshow(winname, S);
}

size_t Matrix::rows() const {
    return _rows;
}

size_t Matrix::cols() const {
    return _cols;
}
