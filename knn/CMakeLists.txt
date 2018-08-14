cmake_minimum_required(VERSION 3.10)
project(knn)

set(CMAKE_CXX_STANDARD 14)

add_executable(knn main.cpp dataset.h functions.h kNN.h kdTree.h)