# ----------------------------------------------------------------------------
#  Minimal Cmake version
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)

set (PROJ_NAME LogPolarFFTTemplateMatching)

project(${PROJ_NAME})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
# ---------------------------------------------------------------------------- 
# We need OpenCV here
# ----------------------------------------------------------------------------
  FIND_PACKAGE(OpenCV)
# ----------------------------------------------------------------------------  
# append sources and heders to the project
# ----------------------------------------------------------------------------
  set(folder_source fftm.cpp test.cpp example.cpp)
  set(folder_header fftm.hpp)
  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})
# ----------------------------------------------------------------------------  
# Add example project
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(${PROJ_NAME}_example fftm.cpp example.cpp fftm.hpp)
# ----------------------------------------------------------------------------
# attaching libraries to out project
# ---------------------------------------------------------------------------- 
  TARGET_LINK_LIBRARIES(${PROJ_NAME}_example ${OpenCV_LIBS})

OPTION(BUILD_TEST "Build test" OFF) 

if(BUILD_TEST)
set(GTEST_ROOT "../googletest/install")
set(GTEST_MSVC_SEARCH "MD")
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
# ----------------------------------------------------------------------------  
# add test executable
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(${PROJ_NAME}_test fftm.cpp test.cpp fftm.hpp)
# ----------------------------------------------------------------------------
# attaching libs to test project
# ---------------------------------------------------------------------------- 
  TARGET_LINK_LIBRARIES(${PROJ_NAME}_test ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})
endif()