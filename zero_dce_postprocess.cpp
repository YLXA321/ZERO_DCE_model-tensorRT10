#include "zero_dce_postprocess.hpp"

namespace zero_dce_postprocess{

    cv::Mat decode_cpu(const float* model_output, const int KInputW, const int KInputH, 
                       const int src_width, const int src_height) {
        cv::Mat src_image;
        if (model_output == nullptr) {
            std::cerr << "ERROR: Model output is null." << std::endl;
            return cv::Mat();
        }
    
        // 创建临时浮点图像（HWC格式，RGB顺序）
        cv::Mat temp_image(KInputH, KInputW, CV_32FC3);
        float* temp_data = reinterpret_cast<float*>(temp_image.data);  // 直接操作内存
    
        // 计算各通道的起始指针
        const int channel_size = KInputH * KInputW;
        const float* r_channel = model_output + 0;           // R通道起始地址
        const float* g_channel = model_output + channel_size; // G通道起始地址
        const float* b_channel = model_output + 2 * channel_size; // B通道起始地址
    
        // 并行化填充（OpenCV自动优化）
        for (int i = 0; i < KInputH; ++i) {
            for (int j = 0; j < KInputW; ++j) {
                const int pixel_idx = (i * KInputW + j) * 3;  // HWC中每个像素的起始位置
                const int ch_idx = i * KInputW + j;          // CHW中当前像素的通道内索引
    
                temp_data[pixel_idx]     = r_channel[ch_idx]; // R
                temp_data[pixel_idx + 1] = g_channel[ch_idx]; // G
                temp_data[pixel_idx + 2] = b_channel[ch_idx]; // B
            }
        }
    
        // 反归一化并转为8UC3（与Python一致）
        temp_image.convertTo(temp_image, CV_8UC3, 255.0);
    
        // Resize到目标尺寸（使用INTER_LINEAR）
        if (KInputW != src_width || KInputH != src_height) {
            cv::resize(temp_image, src_image, cv::Size(src_width, src_height), cv::INTER_LINEAR);
        } else {
            src_image = temp_image.clone();
        }
    
        // RGB转BGR（与Python的cv2.COLOR_RGB2BGR一致）
        cv::cvtColor(src_image, src_image, cv::COLOR_RGB2BGR);
    
        return src_image;
    }
}