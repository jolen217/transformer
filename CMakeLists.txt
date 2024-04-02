# cmake_minimum_require must be the first command of the file
cmake_minimum_required(VERSION 3.10.0)

project(Transformer)

add_executable(transformer main.cpp)

find_package(Eigen3 REQUIRED)
target_include_directories(transformer PRIVATE ${EIGEN3_INCLUDE_DIRS})
