cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME BiCG)
project(${PROJECT_NAME})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/debug)

set(PROJECT_LIBRARY "${PROJECT_NAME}_lib")
set(PROJECT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PROJECT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src")

include_directories("${PROJECT_INCLUDE}")

# OUTPUT DIR
if(CMAKE_BUILD_TYPE STREQUAL Release)
  set(OUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})
else()
  set(OUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG})
endif()

# BUILD
add_subdirectory(src)
add_subdirectory(test)

# REPORT
message(STATUS "")
message(STATUS "General configuration for ${PROJECT_NAME}")
message(STATUS "======================================")
message(STATUS "")
message(STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
