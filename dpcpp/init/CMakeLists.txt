cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER dpcpp)
project(init CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS yes)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(FPGA REQUIRED)

include_directories(../common/include)
add_subdirectory(../../dependencies/fmt fmt)
add_subdirectory(../../dependencies/argparse argparse)
target_compile_options(fmt PUBLIC -Wno-tautological-constant-compare)

add_subdirectory(../../h5read h5read)

fpga_add_executable(init init.cc LINK_LIBRARIES fmt::fmt argparse h5read)
