// #ifndef UTILS_HPP
// #define UTILS_HPP


// // #ifndef CUDA_CHECK
// // #define CUDA_CHECK(callstr)\
// //     {\
// //         cudaError_t error_code = callstr;\
// //         if (error_code != cudaSuccess) {\
// //             std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
// //             assert(0);\
// //         }\
// //     }
// // #endif  // CUDA_CHECK
// #ifndef checkRuntime
// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//     if(code != cudaSuccess){    
//         const char* err_name = cudaGetErrorName(code);    
//         const char* err_message = cudaGetErrorString(code);  
//         printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
//         return false;
//     }
//     return true;
// }
// #endif //checkRuntime



// struct DetectionInfo
// {
//     int class_id{0};
//     float confidence{0.0};
//     cv::Scalar color{};
//     cv::Rect box{};
// };

// #endif


#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// 内联函数定义
inline bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

// 宏定义
#ifndef checkRuntime
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
#endif // checkRuntime

struct DetectionInfo {
    int class_id{0};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

#endif // UTILS_HPP