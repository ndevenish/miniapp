cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER dpcpp)
project(plumbing CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS yes)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(FPGA REQUIRED)

add_subdirectory(../../h5read h5read)
include_directories(../common/include)
add_subdirectory(../../dependencies/fmt fmt)
add_subdirectory(../../dependencies/TinyTIFF TinyTIFF)
add_subdirectory(../../dependencies/argparse argparse)
target_compile_options(fmt PUBLIC -Wno-tautological-constant-compare)


fpga_add_executable(plumbing plumbing.cc kernel.cc
    LINK_LIBRARIES h5read argparse fmt TinyTIFFShared
)
