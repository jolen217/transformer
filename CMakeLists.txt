# cmake_minimum_require must be the first command of the file
cmake_minimum_required(VERSION 3.10.0)

project(Transformer)

add_executable(transformer main.cpp)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
target_include_directories(transformer PRIVATE ${EIGEN3_INCLUDE_DIRS})
