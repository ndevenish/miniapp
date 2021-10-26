cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER dpcpp)
project(mini_stream_test CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS yes)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)


if(NOT DEFINED FPGA_BOARD)
    # Boards are defined as package:board
    # Packages from: "aoc -list-boards" and "aoc -list-board-packages"
    set(FPGA_BOARD "intel_s10sx_pac:pac_s10_usm" CACHE STRING "The package:board combination to pass to aoc")
    message(STATUS "FPGA_BOARD not specified, using default")
endif()
message(STATUS "Configuring for FPGA board: ${FPGA_BOARD}")


########################################################################
########################################################################
# This block is a compressed version of FindFPGA.cmake to make this standalone

# Although - if we have it, the archive-complete-fpga-build feature means
# we can be sure that we don't accidentally delete the fpga artifact.
# Check for it, and make a nullopt function if not present.
include(ArchiveTarget OPTIONAL)
if(NOT COMMAND archive_target)
    function(archive_target)
    endfunction()
endif()

add_custom_target(fpga)
set_target_properties(fpga PROPERTIES EXCLUDE_FROM_ALL yes)
add_custom_target(fpga.report)
set_target_properties(fpga.report PROPERTIES EXCLUDE_FROM_ALL yes)
add_library(FPGA::EMULATOR INTERFACE IMPORTED )
set_target_properties(FPGA::EMULATOR PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "FPGA_EMULATOR;FPGA"
    INTERFACE_COMPILE_OPTIONS "-fintelfpga;${FPGA_WIN_FLAG}"
    INTERFACE_LINK_OPTIONS "-fintelfpga")
add_library(FPGA::FPGA INTERFACE IMPORTED)
set_target_properties(FPGA::FPGA PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "FPGA"
    INTERFACE_COMPILE_OPTIONS "-fintelfpga;${FPGA_WIN_FLAG}"
    INTERFACE_LINK_OPTIONS "-fintelfpga;-Xshardware;-Xsboard=${FPGA_BOARD}")
## Convenience function to add a target and variants at the same time
function(fpga_add_executable name)
    cmake_parse_arguments(PARSE_ARGV 1 _addexec "" "" "LINK_LIBRARIES")
    add_executable(${name}.fpga_emu ${_addexec_UNPARSED_ARGUMENTS})
    target_link_libraries(${name}.fpga_emu FPGA::EMULATOR ${NAME} ${_addexec_LINK_LIBRARIES})

    add_executable(${name}_report.a ${_addexec_UNPARSED_ARGUMENTS})
    target_link_libraries(${name}_report.a FPGA::FPGA ${NAME} ${_addexec_LINK_LIBRARIES})
    target_link_options(${name}_report.a PRIVATE "-fsycl-link=early")
    # By default, don't make the report, unless requested
    set_target_properties(${name}_report.a PROPERTIES EXCLUDE_FROM_ALL yes)
    add_dependencies(fpga.report ${name}_report.a)

    add_executable(${name}.fpga ${_addexec_UNPARSED_ARGUMENTS})
    target_link_libraries(${name}.fpga FPGA::FPGA ${NAME} ${_addexec_LINK_LIBRARIES})
    set_target_properties(${name}.fpga PROPERTIES EXCLUDE_FROM_ALL yes)
    set_property(TARGET ${name}.fpga PROPERTY JOB_POOL_LINK console)
    add_dependencies(fpga ${name}.fpga)
    archive_target(${name}.fpga)
endfunction()
# End of inline FPGA convenience
########################################################################
########################################################################

fpga_add_executable(mini_stream_test ministream.cc)
