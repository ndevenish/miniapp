cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER icpx)
project(enumerate CXX)

add_executable(enumerate enumerate.cc)
