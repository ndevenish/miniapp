cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER dpcpp)
project(reduced CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS yes)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(FPGA REQUIRED)

fpga_add_executable(reduced reduced.cc)
