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

void bimshow(const String &winname, Mat &I);

void hishow(const String &winname, Mat &I, int hist_size = 256, float lo = 0, float hi = 256, bool uniform = true,
            bool accumulate = false);

std::vector<Point> generatePoints(Mat &I, size_t size, bool grid);

class Vector {
private:
    size_t _size;
    std::vector<double> _data;

public:
    Vector(const std::vector<double> &vector);

    Vector(size_t size);

    static double distance(const Vector &a, const Vector &b) {
        double result = 0;
        for (size_t i = 0; i < a._size; ++i) {
            result += pow(a[i] - b[i], 2);
        }
        return sqrt(result);
    }

    std::vector<double> unwrap() const;

    Vector normalize() const;

    size_t argmax() const;

    Vector operator+(const Vector &b) const;

    Vector &operator+=(const Vector &b);

    Vector operator/(const double &b) const;

    Vector &operator/=(const double &b);

    double &operator[](size_t i);

    double operator[](size_t i) const;

    bool operator==(const Vector &rhs) const;

    friend std::ostream &operator<<(std::ostream &os, const Vector &row);

    size_t size() const;
};

class Matrix {
private:
    size_t _rows;
    size_t _cols;
    std::vector<double> _data;

public:
    Matrix();

    Matrix(size_t rows, size_t cols);

    Matrix(const Matrix &matrix);

    double &operator()(size_t i, size_t j);

    double operator()(size_t i, size_t j) const;

    Vector getRow(size_t n) const;

    void addRow(Vector row);

    int rowPosition(Vector row);

    Vector getCol(size_t n) const;

    void addCol(Vector col);

    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);

    size_t rows() const;

    size_t cols() const;
};

#endif //TABLEWARE_RECOGNITION_UTIL_H
