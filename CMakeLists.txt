cmake_minimum_required(VERSION 3.12)
project(tableware-recognition)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS filesystem unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})


add_library(improc src/processing.cpp src/util.cpp src/util.h src/feature_extraction.cpp src/feature_extraction.h src/segmentation.cpp src/segmentation.h src/classifier.cpp src/classifier.h)
target_link_libraries(improc ${OpenCV_LIBS})
target_link_libraries(improc ${Boost_LIBRARIES})

add_executable(recognize src/main.cpp)
target_link_libraries(recognize improc ${Boost_LIBRARIES})

add_executable(unit-tests src/test.cpp)
target_link_libraries(unit-tests improc)
