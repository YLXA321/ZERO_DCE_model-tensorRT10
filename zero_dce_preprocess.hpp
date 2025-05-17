#ifndef ZERO_DCE_PREPROCESS_HPP
#define ZERO_DCE_PREPROCESS_HPP


#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cuda_runtime.h>
// #include <nppi_geometry.h>

namespace zero_dce_preprocess{


void preprocess_gpu(const cv::Mat& srcImg, float* dstDevData, const int dstHeight, const int dstWidth, cudaStream_t stream);
/*
srcImg:     source image for inference
dstDevData: data after preprocess (resize / bgr to rgb / hwc to chw / normalize)
dstHeight:  CNN input height
dstWidth:   CNN input width
*/


void preprocess_cpu(cv::Mat &srcImg, float* dstDevData, const int width, const int height);


} // zero_dce_preprocess

#endif  // PREPROCESS_HPP
