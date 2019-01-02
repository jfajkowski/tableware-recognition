cmake_minimum_required(VERSION 3.12)
project(tableware-recognition)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_executable(tableware-recognition src/main.cpp src/processing.cpp src/util.cpp src/util.h src/feature_extraction.cpp src/feature_extraction.h src/segmentation.cpp src/segmentation.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(tableware-recognition ${OpenCV_LIBS})
