cmake_minimum_required(VERSION 3.19)

project(baseline)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_SOURCE_DIR}/../cmake/Modules")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 99)
set(CMAKE_BUILD_TYPE Debug)

#include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

#IF(CMAKE_COMPILER_IS_GNUCC)
#    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
#ENDIF(CMAKE_COMPILER_IS_GNUCC)

find_package(Boost REQUIRED)
find_package(Dials REQUIRED)
find_package(HDF5 REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenMP)

add_subdirectory(../h5read h5read)

add_library(baseline SHARED baseline.cpp )
if(OpenMP_CXX_FOUND)
    message("Found OpenMP")
    target_link_libraries(baseline PUBLIC Dials::Dials benchmark::benchmark h5read OpenMP::OpenMP_CXX)
else()
    message("Didn't find OpenMP")
    target_link_libraries(baseline PUBLIC Dials::Dials benchmark::benchmark h5read)
endif()

add_executable(bm bm.cpp)
target_link_libraries(bm baseline)

add_executable(miniapp miniapp.c)
target_link_libraries(miniapp PUBLIC hdf5::hdf5 baseline h5read OpenMP::OpenMP_C)
