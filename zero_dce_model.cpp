#include "zero_dce_model.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <sys/stat.h>

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

ZeroDCEModel::ZeroDCEModel(std::string onnxfilepath, bool fp16, int maxbatch, bool usegpu)
:onnx_file_path{onnxfilepath}, FP16(fp16), maxBatch(maxbatch), useGPU(usegpu)
{
    // auto level = logger::Level::FATAL;
    // auto level = logger::Level::ERROR;
    // auto level = logger::Level::WARN;
    // auto level = logger::Level::INFO;
    // auto level = logger::Level::VERB;
    auto level = trtlogger::Level::DEBUG;
    auto idx = onnx_file_path.find(".onnx");
    auto basename = onnx_file_path.substr(0, idx);
    m_enginePath = basename + ".engine";
    if (file_exists(m_enginePath)){
        std::cout << "start building model from engine file: " << m_enginePath;
        this->Runtime(m_enginePath,level, 1);
    }else{
        std::cout << "start building model from onnx file: " << onnx_file_path;
        
        build_engine::genEngine(onnx_file_path, m_enginePath, level, maxbatch);
        this->Runtime(m_enginePath, level, 1);
    }
    this->trtIOMemory();
}


bool ZeroDCEModel::Runtime(std::string engine_file_path, trtlogger::Logger level,int maxBatch){

    auto logger = std::make_shared<trtlogger::Logger>(level);

   // 初始化trt插件
//    initLibNvInferPlugins(&logger, "");
    
    std::ifstream engineFile(engine_file_path, std::ios::binary);
    long int fsize = 0;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineString(fsize);
    engineFile.read(engineString.data(), fsize);
    if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return false; }

    // 创建推理引擎
    m_runtime.reset(nvinfer1::createInferRuntime(*logger));
    if(!m_runtime){
        std::cout<<" (T_T)~~~, Failed to create runtime."<<std::endl;
        return false;
    }

    // 反序列化推理引擎
    m_engine.reset(m_runtime->deserializeCudaEngine(engineString.data(), fsize));
    if(!m_engine){
        std::cout<<" (T_T)~~~, Failed to deserialize."<<std::endl;
        return false;
    }

    // 获取优化后的模型的输入维度和输出维度
    // int nbBindings = m_engine->getNbBindings(); // trt8.5 以前版本
    int nbBindings = m_engine->getNbIOTensors();  // trt8.5 以后版本

    // 推理执行上下文
    m_context.reset(m_engine->createExecutionContext());
    if(!m_context){
        std::cout<<" (T_T)~~~, Failed to create ExecutionContext."<<std::endl;
        return false;
    }

    auto input_dims = m_context->getTensorShape("input");
    input_dims.d[0] = maxBatch;
    m_context->setInputShape("input", input_dims);

    std::cout << " ~~Congratulations! 🎉🎉🎉~  create execution context success!!! ✨✨✨~~ " << std::endl;
    return true; 
}


bool ZeroDCEModel::trtIOMemory() {

    m_inputDims = m_context->getTensorShape("input"); // 模型输入
    m_outputDims[0] = m_context->getTensorShape("output1"); //第一个输出
    m_outputDims[1] = m_context->getTensorShape("output2"); //第二个输出
    m_outputDims[2] = m_context->getTensorShape("output3"); //第三个输出

    this->kInputH = m_inputDims.d[2];
    this->kInputW = m_inputDims.d[3];
    
    m_inputSize = m_inputDims.d[0] * m_inputDims.d[1] * m_inputDims.d[2] * m_inputDims.d[3] * sizeof(float);
    
    m_outputSize[0] = m_outputDims[0].d[0] * m_outputDims[0].d[1] * m_outputDims[0].d[2] * m_outputDims[0].d[3] * sizeof(float);
    m_outputSize[1] = m_outputDims[1].d[0] * m_outputDims[1].d[1] * m_outputDims[1].d[2] * m_outputDims[1].d[3] * sizeof(float);
    m_outputSize[2] = m_outputDims[2].d[0] * m_outputDims[2].d[1] * m_outputDims[2].d[2] * m_outputDims[2].d[3] * sizeof(float);
    
    // 声明cuda的内存大小
    checkRuntime(cudaMalloc(&buffers[0], m_inputSize));
    checkRuntime(cudaMalloc(&buffers[1], m_outputSize[0]));
    checkRuntime(cudaMalloc(&buffers[2], m_outputSize[1]));
    checkRuntime(cudaMalloc(&buffers[3], m_outputSize[2]));


    // 声明cpu内存大小
    checkRuntime(cudaMallocHost(&cpu_buffers[0], m_inputSize));
    checkRuntime(cudaMallocHost(&cpu_buffers[1], m_outputSize[0]));
    checkRuntime(cudaMallocHost(&cpu_buffers[2], m_outputSize[1]));
    checkRuntime(cudaMallocHost(&cpu_buffers[3], m_outputSize[2]));
   
    m_context->setTensorAddress("input", buffers[0]);
    m_context->setTensorAddress("output1", buffers[1]);
    m_context->setTensorAddress("output2", buffers[2]);
    m_context->setTensorAddress("output3", buffers[3]);

    checkRuntime(cudaStreamCreate(&m_stream));

    return true; 
}


cv::Mat ZeroDCEModel::doInference(cv::Mat& frame) {

    if(useGPU){
        zero_dce_preprocess::preprocess_gpu(frame, (float*)buffers[0], kInputH, kInputW,  m_stream);
    }else{
        zero_dce_preprocess::preprocess_cpu(frame, cpu_buffers[0], kInputW, kInputH);
        // Preprocess -- 将host的数据移动到device上
        checkRuntime(cudaMemcpyAsync(buffers[0], cpu_buffers[0], m_inputSize, cudaMemcpyHostToDevice, m_stream));
    }
    
    bool status = this->m_context->enqueueV3(m_stream);
    if (!status) std::cerr << "(T_T)~~~, Failed to create ExecutionContext." << std::endl;

    // 将gpu推理的结果返回到cpu上面处理
    checkRuntime(cudaMemcpyAsync(cpu_buffers[1], buffers[1], m_outputSize[0], cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(cudaMemcpyAsync(cpu_buffers[2], buffers[2], m_outputSize[1], cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(cudaMemcpyAsync(cpu_buffers[3], buffers[3], m_outputSize[2], cudaMemcpyDeviceToHost, m_stream));
    
    checkRuntime(cudaStreamSynchronize(m_stream));

    int height = frame.rows;
    int width = frame.cols;
    cv::Mat enhance_image;

    if(useGPU){
        enhance_image = zero_dce_postprocess::decode_gpu(buffers[2],kInputW,kInputH,width,height);
    }else{
        // cv::Mat enhance_image_1 = zero_dce_postprocess::decode_cpu(cpu_buffers[1],kInputW,kInputH,height,width);
        enhance_image = zero_dce_postprocess::decode_cpu(cpu_buffers[2],kInputW,kInputH,width,height);
        // cv::Mat r = zero_dce_postprocess::decode_cpu(cpu_buffers[3],kInputW,kInputH,height,width);

    }
    

    return enhance_image;

}

ZeroDCEModel::~ZeroDCEModel() 
{
    // 销毁 CUDA 流
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }

    // 释放设备内存
    for (auto& buffer : buffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }

    // 释放主机内存
    for (auto& cpu_buffer : cpu_buffers) {
        if (cpu_buffer) {
            cudaFree(cpu_buffer);
            cpu_buffer = nullptr;
        }
    }
}
