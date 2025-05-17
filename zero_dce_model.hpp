#ifndef ZERO_DCE_HPP
#define ZERO_DCE_HPP

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include "TrtLogger.hpp"
#include "genEngine.hpp"
#include "zero_dce_preprocess.hpp"
#include "zero_dce_postprocess.hpp"


class ZeroDCEModel
{
public:

    ZeroDCEModel(std::string onnxfilepath, bool fp16, int maxbatch, bool usegpu);
    ~ZeroDCEModel();
    cv::Mat doInference(cv::Mat& frame);             /*检测图片*/

private:

    bool trtIOMemory();
	bool Runtime(std::string engine_file_path, trtlogger::Logger level, int maxBatch);                                                 /*从engine穿件推理运行时，执行上下文*/
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;                   /*声明模型的推理运行时指针*/
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;                 /*声明模型反序列化指针*/
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;          /*声明模型执行上下文指针*/

    float* buffers[4] {nullptr};                                            /*为模型申请输入输出的缓冲区地址*/
    float* cpu_buffers[4] {nullptr};                                        /*声明输入输出缓冲区地址*/
   
    nvinfer1::Dims m_inputDims {};                                      /*声明输入图片属性的索引*/
    nvinfer1::Dims m_outputDims[3] {};                                    /*声明输出图片属性的索引*/
    cudaStream_t   m_stream;                                        /*声明cuda流*/

    std::string onnx_file_path {};                                  /*指定输入的onnx模型的地址*/
    std::string m_enginePath {};                                    /*指定生成的engine模型的地址*/
    bool FP16 {true};                                               /* 判断是否使用半精度进行面模型优化*/
    int m_inputSize {};                                             /*图像需要预处理的大小*/
    int m_outputSize[3] {};                                            /*输入图像的大小*/
    bool useGPU {false};
    int kInputH {};
    int kInputW {};
    int maxBatch {1};
};

#endif // ZERO_DCE_HPP

