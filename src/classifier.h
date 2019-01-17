//
// Created by jfajkowski on 17.01.19.
//

#ifndef TABLEWARE_RECOGNITION_CLASSIFIER_H
#define TABLEWARE_RECOGNITION_CLASSIFIER_H

#include "util.h"

class Classifier {
private:
    Matrix model;

public:
    void fit(const Matrix &X, const Matrix &y);

    Matrix predict(const Matrix &X);
};


#endif //TABLEWARE_RECOGNITION_CLASSIFIER_H
