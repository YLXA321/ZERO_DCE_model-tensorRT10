#include "zero_dce_preprocess.hpp"

namespace zero_dce_preprocess{

// CUDA核函数：BGR转RGB + HWC→CHW + 归一化
__global__ void bgr2rgb_planar_kernel(
    const uchar* src, 
    float* dst, 
    int width, 
    int height,
    float scale
) {
    // 计算像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 计算HWC索引
    const int src_idx = (y * width + x) * 3;

    // 读取BGR值
    uchar b = src[src_idx];
    uchar g = src[src_idx + 1];
    uchar r = src[src_idx + 2];

    // 计算CHW目标索引
    const int plane_size = width * height;
    const int dst_r = y * width + x;                // R通道
    const int dst_g = dst_r + plane_size;           // G通道
    const int dst_b = dst_g + plane_size;           // B通道

    // 写入目标内存（归一化）
    dst[dst_r] = static_cast<float>(r) * scale;
    dst[dst_g] = static_cast<float>(g) * scale;
    dst[dst_b] = static_cast<float>(b) * scale;
}

void preprocess_gpu(
    const cv::Mat& src_img,
    float* gpu_dst,      // 预分配的GPU内存（CHW格式）
    int target_height,
    int target_width,
    cudaStream_t stream = 0
) {
    // 参数检查
    if (src_img.empty()) {
        throw std::runtime_error("src_img is empty");
    }
    if (!gpu_dst) {
        throw std::runtime_error("gpu_dst is nullptr");
    }

    // 设备内存指针
    uchar* d_src = nullptr;
    uchar* d_resized = nullptr;

    // 1. 分配临时设备内存
    const size_t src_size = src_img.total() * src_img.elemSize();
    cudaMallocAsync(&d_src, src_size, stream);
    cudaMemcpyAsync(d_src, src_img.data, src_size, 
                   cudaMemcpyHostToDevice, stream);

    // 2. 执行Resize（仅在尺寸不匹配时）
    const bool need_resize = (src_img.rows != target_height) || 
                            (src_img.cols != target_width);
    if (need_resize) {
        // 分配缩放临时缓冲区
        const size_t resize_size = target_width * target_height * 3;
        cudaMallocAsync(&d_resized, resize_size, stream);

        // 配置NPP参数
        NppiSize src_size_nppi = {src_img.cols, src_img.rows};
        NppiRect src_roi = {0, 0, src_img.cols, src_img.rows};
        NppiSize dst_size = {target_width, target_height};
        NppiRect dst_roi = {0, 0, target_width, target_height};

        // 执行Resize（异步，指定stream）
        NppStatus status = nppiResize_8u_C3R(
            d_src, 
            static_cast<int>(src_img.step),  // 源图像步长（每行字节数）
            src_size_nppi,                   // 源图像尺寸
            src_roi,
            d_resized, 
            target_width * 3,                // 目标图像步长
            dst_size, 
            dst_roi,
            NPPI_INTER_LINEAR                // 使用可用的插值方法
        );
        
        if (status != NPP_SUCCESS) {
            throw std::runtime_error("nppiResize_8u_C3R failed with error code: " + std::to_string(status));
        }
    } else {
        d_resized = d_src;  // 直接使用原始数据
    }

    // 3. 启动预处理核函数
    const dim3 block(32, 8);
    const dim3 grid(
        (target_width + block.x - 1) / block.x,
        (target_height + block.y - 1) / block.y
    );

    // 计算归一化系数（1/255）
    const float scale = 1.0f / 255.0f;

    bgr2rgb_planar_kernel<<<grid, block, 0, stream>>>(
        need_resize ? d_resized : d_src,
        gpu_dst,
        target_width,
        target_height,
        scale
    );

    // 4. 释放临时内存（异步）
    if (d_src) cudaFreeAsync(d_src, stream);
    if (need_resize && d_resized) cudaFreeAsync(d_resized, stream);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} //zero_dce_preprocess