#include "zero_dce_preprocess.hpp"

namespace zero_dce_preprocess{
    
/// @brief 
/// @param srcImg 输入的原始图像 
/// @param dstDevData 输出的图像的地址
/// @param width 模型接收的宽
/// @param height 模型接收的高

void preprocess_cpu(cv::Mat &srcImg, float* dstDevData, const int width, const int height) {
    if (srcImg.data == nullptr) {
        std::cerr << "ERROR: Image file not found! Program terminated" << std::endl;
        return;
    }

    cv::Mat dstimg;
    if (srcImg.rows != height || srcImg.cols != width) {
        cv::resize(srcImg, dstimg, cv::Size(width, height), cv::INTER_AREA);
    } else {
        dstimg = srcImg.clone();
    }

    // BGR→RGB转换 + HWC→CHW转换
    int index = 0;
    int offset_ch0 = width * height * 0;  // R通道
    int offset_ch1 = width * height * 1;  // G通道
    int offset_ch2 = width * height * 2;  // B通道
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            index = i * width * 3 + j * 3;
            // 从BGR数据中提取并赋值到目标通道
            dstDevData[offset_ch0++] = dstimg.data[index + 2] / 255.0f;  // R
            dstDevData[offset_ch1++] = dstimg.data[index + 1] / 255.0f;  // G
            dstDevData[offset_ch2++] = dstimg.data[index + 0] / 255.0f;  // B
        }
    }
}

} // zero_dce_process