# ----------------------------------------------------------------------------
# минимальна€ верси€ CMake, способна€ переварить этот файл
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)

set (PROJ_NAME LogPolarFFTTemplateMatching)

project(${PROJ_NAME})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
# ---------------------------------------------------------------------------- 
# ѕредполагаем, что FindOpenCV.cmake расположен по адресу CMAKE_MODULE_PATH.
# ----------------------------------------------------------------------------
  FIND_PACKAGE(OpenCV)
# ----------------------------------------------------------------------------  
# ищем все .cpp и .h файлы и добавл€ем их в наш проект 
# ----------------------------------------------------------------------------
  set(folder_source main.cpp)
  set(folder_header  )
  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})
# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(${PROJ_NAME} ${folder_source} ${folder_header})
# ----------------------------------------------------------------------------
# добавл€ем Lib-ы opencv 
# ---------------------------------------------------------------------------- 
  TARGET_LINK_LIBRARIES(${PROJ_NAME} ${OpenCV_LIBS})
