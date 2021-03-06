cmake_minimum_required(VERSION 3.19)

project(baseline)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_SOURCE_DIR}/../cmake/Modules")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 99)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(Boost REQUIRED)
find_package(Dials REQUIRED)
find_package(HDF5 REQUIRED)
find_package(benchmark REQUIRED)

add_subdirectory(../h5read h5read)

add_library(baseline SHARED baseline.cpp )
target_link_libraries(baseline PUBLIC Dials::Dials benchmark::benchmark h5read)

add_executable(bm bm.cpp)
target_link_libraries(bm baseline)

add_executable(miniapp miniapp.c)
target_link_libraries(miniapp PUBLIC hdf5::hdf5 baseline h5read)
