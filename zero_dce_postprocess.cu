#include "zero_dce_postprocess.hpp"


namespace zero_dce_postprocess {

// CUDA核函数：将CHW格式的浮点数据转换为HWC格式的uchar数据（包含反归一化）
__global__ void chw_to_hwc_rgb_kernel(
    const float* __restrict__ input,  // 输入数据（CHW格式，归一化的浮点数）
    uchar* __restrict__ output,       // 输出数据（HWC格式，RGB顺序，非归一化的uchar）
    int width,                        // 图像宽度
    int height,                       // 图像高度
    float scale                       // 反归一化因子（通常为255.0）
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // CHW格式的通道位置
    const int channel_size = width * height;
    const float r_val = input[0 * channel_size + y * width + x];
    const float g_val = input[1 * channel_size + y * width + x];
    const float b_val = input[2 * channel_size + y * width + x];
    
    // 反归一化并写入HWC格式
    int hwc_idx = (y * width + x) * 3;
    output[hwc_idx + 0] = static_cast<uchar>(r_val * scale);
    output[hwc_idx + 1] = static_cast<uchar>(g_val * scale);
    output[hwc_idx + 2] = static_cast<uchar>(b_val * scale);
}

// 优化版CUDA核函数：使用共享内存提高内存访问效率
__global__ void chw_to_hwc_rgb_kernel_optimized(
    const float* __restrict__ input,
    uchar* __restrict__ output,
    int width,
    int height,
    float scale
) {
    // 使用共享内存缓存输入数据，提高内存访问效率
    __shared__ float tile[32][32][3];  // 假设block大小为32x32
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算全局坐标
    int x = bx * 32 + tx;
    int y = by * 32 + ty;
    
    const int channel_size = width * height;
    
    // 加载数据到共享内存
    if (x < width && y < height) {
        int idx = y * width + x;
        tile[ty][tx][0] = input[0 * channel_size + idx];
        tile[ty][tx][1] = input[1 * channel_size + idx];
        tile[ty][tx][2] = input[2 * channel_size + idx];
    }
    
    __syncthreads();
    
    // 将共享内存中的数据写入输出
    if (x < width && y < height) {
        int hwc_idx = (y * width + x) * 3;
        output[hwc_idx + 0] = static_cast<uchar>(tile[ty][tx][0] * scale);
        output[hwc_idx + 1] = static_cast<uchar>(tile[ty][tx][1] * scale);
        output[hwc_idx + 2] = static_cast<uchar>(tile[ty][tx][2] * scale);
    }
}

// CUDA版本的后处理函数（优化版）
cv::Mat decode_gpu(const float* model_output, const int KInputW, const int KInputH, 
                   const int src_width, const int src_height) {
    if (model_output == nullptr) {
        std::cerr << "ERROR: Model output is null." << std::endl;
        return cv::Mat();
    }
    
    // 1. 分配GPU内存
    uchar* d_rgb = nullptr;
    size_t rgb_size = KInputW * KInputH * 3 * sizeof(uchar);
    cudaMalloc(&d_rgb, rgb_size);
    
    // 2. 执行CHW到HWC的转换和反归一化
    // 使用优化版核函数
    dim3 block_size(32, 32);
    dim3 grid_size((KInputW + block_size.x - 1) / block_size.x, 
                  (KInputH + block_size.y - 1) / block_size.y);
    
    chw_to_hwc_rgb_kernel_optimized<<<grid_size, block_size>>>(
        model_output, d_rgb, KInputW, KInputH, 255.0f
    );
    
    // 3. 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_rgb);
        return cv::Mat();
    }
    
    // 4. 执行GPU端的图像缩放（如果需要）
    uchar* d_resized = nullptr;
    cv::Mat src_image;
    
    if (KInputW != src_width || KInputH != src_height) {
        size_t resized_size = src_width * src_height * 3 * sizeof(uchar);
        cudaMalloc(&d_resized, resized_size);
        
        // 使用NPP库进行图像缩放
        NppiSize src_size = {KInputW, KInputH};
        NppiSize dst_size = {src_width, src_height};
        NppiRect src_roi = {0, 0, KInputW, KInputH};
        NppiRect dst_roi = {0, 0, src_width, src_height};

        // 执行Resize操作（修正版）
        NppStatus status = nppiResize_8u_C3R(
            d_rgb,                // 源图像数据
            KInputW * 3,          // 源图像步长（int类型）
            src_size,             // 源图像尺寸（NppiSize类型）
            src_roi,              // 源图像ROI
            d_resized,            // 目标图像数据
            src_width * 3,        // 目标图像步长（int类型）
            dst_size,             // 目标图像尺寸（NppiSize类型）
            dst_roi,              // 目标图像ROI
            NPPI_INTER_LINEAR     // 插值方法
        );

       
        if (status != NPP_SUCCESS) {
            std::cerr << "NPP resize failed with error: " << status << std::endl;
            cudaFree(d_rgb);
            cudaFree(d_resized);
            return cv::Mat();
        }
        
        // 为最终结果分配内存
        src_image = cv::Mat(src_height, src_width, CV_8UC3);
        
        // 从GPU复制缩放后的结果
        cudaMemcpy(src_image.data, d_resized, resized_size, cudaMemcpyDeviceToHost);
        
        // 释放缩放后的内存
        cudaFree(d_resized);
    } else {
        // 不需要缩放，直接复制
        src_image = cv::Mat(KInputH, KInputW, CV_8UC3);
        cudaMemcpy(src_image.data, d_rgb, rgb_size, cudaMemcpyDeviceToHost);
    }
    
    // 5. 释放GPU内存
    cudaFree(d_rgb);
    
    // 6. RGB转BGR
    cv::cvtColor(src_image, src_image, cv::COLOR_RGB2BGR);
    
    return src_image;
}


cv::Mat decode_cpu(float* model_output, const int KInputW, const int KInputH, 
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
        cv::resize(temp_image, src_image, cv::Size(src_height, src_width), cv::INTER_LINEAR);
    } else {
        src_image = temp_image.clone();
    }
    
    // RGB转BGR（与Python的cv2.COLOR_RGB2BGR一致）
    cv::cvtColor(src_image, src_image, cv::COLOR_RGB2BGR);
    
    return src_image;
}

} // namespace zero_dce_postprocess