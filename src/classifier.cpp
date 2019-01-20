//
// Created by jfajkowski on 17.01.19.
//

#include "classifier.h"

void Classifier::fit(const Matrix &X, const Matrix &y) {
    Matrix y_distinct;
    std::map<size_t, Vector> index_to_feature_vector;
    std::map<size_t, double> index_to_example_counter;

    for (size_t i = 0; i < y.rows(); ++i) {
        Vector y_row = y.getRow(i);
        if (y_distinct.rowPosition(y_row) == -1) {
            y_distinct.addRow(y_row);
            size_t index = y_distinct.rows() - 1;
            index_to_feature_vector.emplace(index, Vector(X.cols()));
            index_to_example_counter.emplace(index, 0);
        }
    }

    for (size_t i = 0; i < X.rows(); ++i) {
        size_t index = static_cast<size_t>(y_distinct.rowPosition(y.getRow(i)));
        index_to_feature_vector.at(index) += X.getRow(i);
        index_to_example_counter.at(index) += 1;
    }

    for (size_t i = 0; i < y_distinct.rows(); ++i) {
        _model_X.addRow(index_to_feature_vector.at(i) / index_to_example_counter.at(i));
        _model_y.addRow(y_distinct.getRow(i));
    }
}

Matrix Classifier::predict_proba(const Matrix &X) {
    Matrix result;
    for (size_t i = 0; i < X.rows(); ++i) {
        std::vector<double> data(_model_y.cols());
        for (size_t j = 0; j < _model_y.rows(); ++j) {
            size_t class_index = _model_y.getRow(j).argmax();
            data[class_index] = (1 / (1 + Vector::distance(_model_X.getRow(j), X.getRow(i))));
        }
        result.addRow(Vector(data));
    }
    return result;
}

Matrix Classifier::predict(const Matrix &X) {
    Matrix result;
    Matrix proba_vectors = predict_proba(X);
    for (size_t i = 0; i < proba_vectors.rows(); ++i) {
        auto proba_vector = proba_vectors.getRow(i);
        size_t max_index = proba_vector.argmax();
        for (size_t j = 0; j < proba_vector.size(); ++j) {
            proba_vector[j] = j == max_index ? 1 : 0;
        }
        result.addRow(proba_vector);
    }
    return result;
}

double Classifier::accuracy(const Matrix &X, const Matrix &y_true) {
    Matrix y_pred = predict(X);
    double correct = 0.0;
    for (size_t i = 0; i < y_pred.rows(); ++i) {
        correct += y_pred.getRow(i) == y_true.getRow(i) ? 1 : 0;
    }
    return correct / y_pred.rows();
}
