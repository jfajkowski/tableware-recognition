//
// Created by jfajkowski on 17.01.19.
//

#ifndef TABLEWARE_RECOGNITION_CLASSIFIER_H
#define TABLEWARE_RECOGNITION_CLASSIFIER_H

#include "util.h"

class Classifier {
private:
    std::vector<double> means;
    std::vector<double> variances;
    Matrix _model_X;
    Matrix _model_y;

public:
    void fit(const Matrix &X, const Matrix &y);

    Matrix predict_proba(const Matrix &X);

    Matrix predict(const Matrix &X);

    double accuracy(const Matrix &X, const Matrix &y_true);
};


#endif //TABLEWARE_RECOGNITION_CLASSIFIER_H
