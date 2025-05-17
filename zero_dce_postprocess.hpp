#ifndef ZERO_DCE_POSTPROCESS_HPP
#define ZERO_DCE_POSTPROCESS_HPP

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nppi.h>
#include "utils.hpp"
#include <random>


namespace zero_dce_postprocess{



    cv::Mat decode_gpu(const float* model_output, const int KInputW, const int KInputH, 
                        const int src_width, const int src_height);


    cv::Mat decode_cpu( const float* model_output, const int KInputW, const int KInputH, 
                        const int src_width, const int src_heighe );

}
#endif  // ZERO_DCE_POSTPROCESS_HPP