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
add_subdirectory(../../dependencies/fmt fmt)
# add_subdirectory(../common common)
include_directories(../common/include)

fpga_add_executable(plumbing plumbing.cc
    LINK_LIBRARIES h5read fmt::fmt-header-only
)
target_link_options(plumbing_report.a PRIVATE -Xsdont-error-if-large-area-est)