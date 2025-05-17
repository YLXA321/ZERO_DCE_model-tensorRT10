
cmake_minimum_required(VERSION 3.11)
project(obb LANGUAGES CXX)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-12.3/include)
link_directories(/usr/local/cuda-12.3/lib64)

# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
# include_directories(/opt/TensorRT/TensorRT-10.8.0.43/include/)
# link_directories(/opt/TensorRT/TensorRT-10.8.0.43/lib/)
include_directories(/opt/TensorRT/TensorRT-10.9.0.34/include/)
link_directories(/opt/TensorRT/TensorRT-10.9.0.34/lib/)


# OpenCV
# include_directories(/opt/opencv4.7/opencv-4.7.0/include/)
# include_directories(/usr/local/include/opencv4)
# link_directories(/usr/local/lib)
find_package(OpenCV REQUIRED)

# 包含目录设置
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})

# 添加源文件并创建可执行文件
add_executable(build
    main12.cpp
    TrtLogger.cpp
    genEngine.cu
    # TrtModel.cpp
    # preprocess.cu 
    # preprocess.cpp 
    # detect_postprocess.cu 
    # detect_postprocess.cpp 
    zero_dce_model.cpp
    zero_dce_postprocess.cpp
    zero_dce_postprocess.cu
    zero_dce_preprocess.cpp
    zero_dce_preprocess.cu
)

# 设置目标属性（如CUDA架构）
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# 链接OpenCV库
target_link_libraries(build PRIVATE 
                        ${OpenCV_LIBS} 

                        ${TensorRT_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser

                        ${CUDA_LIBRARIES} nppig nppc cudart
                        
                        )

# # 链接TensorRT库
# target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser)
# target_link_libraries(build PRIVATE ${CUDA_LIBRARIES} cudart)

# 如果使用较新的CMake版本，可以考虑使用以下方式链接：
# target_link_libraries(build PRIVATE OpenCV::OpenCV TensorRT::TensorRT)
