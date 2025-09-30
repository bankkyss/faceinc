cmake_minimum_required(VERSION 3.15)
project(ai_face_reg_core_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui dnn)

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Please provide -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}" "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64")
if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not locate onnxruntime library under ${ONNXRUNTIME_ROOT}")
endif()

add_executable(face_pipeline
    FacePipeline.cpp
    AdaFaceEmbedder.cpp
    ../alignment_SCRFD/SCRFD.cpp
)

target_include_directories(face_pipeline PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(face_pipeline PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

if(DEFINED USE_CUDA AND USE_CUDA)
    target_compile_definitions(face_pipeline PRIVATE USE_CUDA=1)
endif()
