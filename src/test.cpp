//
// Created by jfajkowski on 19.01.19.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE tests

#include <boost/test/unit_test.hpp>
#include "classifier.h"

BOOST_AUTO_TEST_CASE(row_test) {
    Vector a(std::vector<double>({10.0, 0.0, 0.0}));
    Vector b(std::vector<double>({0.0, 10.0, 0.0}));

    Vector expected = Vector(std::vector<double>({10.0, 10.0, 0.0}));
    Vector actual = a + b;
    BOOST_CHECK(actual == expected);

    expected = Vector(std::vector<double>({10.0, 10.0, 10.0}));
    actual += Vector(std::vector<double>({0.0, 0.0, 10.0}));
    BOOST_CHECK(actual == expected);

    expected = Vector(std::vector<double>({1.0, 1.0, 1.0}));
    actual /= 10.0;
    BOOST_CHECK(actual == expected);

    expected = Vector(std::vector<double>({1.0, 0.0, 0.0}));
    actual = a / 10.0;
    BOOST_CHECK(actual == expected);
}

BOOST_AUTO_TEST_CASE(matrix_test) {
    Matrix m;
    m.addCol(std::vector<double>({1.0, 0.0, 0.0}));
    m.addCol(std::vector<double>({0.0, 0.0, 1.0}));
    std::cout << m << std::endl;
    std::cout << m.getCol(0).normalize() << std::endl;
}

BOOST_AUTO_TEST_CASE(classifier_test) {
    Matrix X;
    X.addRow(std::vector<double>({1.0, 0.0}));
    X.addRow(std::vector<double>({0.0, 1.0}));

    Matrix y_true;
    y_true.addRow(std::vector<double>({1.0, 0.0}));
    y_true.addRow(std::vector<double>({0.0, 1.0}));

    Classifier classifier;
    classifier.fit(X, y_true);
    Matrix y_pred = classifier.predict(X);

    for (size_t i = 0; i < 2; ++i) {
        BOOST_CHECK(y_pred.getRow(i) == y_true.getRow(i));
    }
}