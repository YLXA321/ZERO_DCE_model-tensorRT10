#include "genEngine.hpp"

namespace build_engine{

bool genEngine(std::string onnx_file_path, std::string save_engine_path, trtlogger::Logger level, int maxbatch){

    auto logger = std::make_shared<trtlogger::Logger>(level);

    // 创建builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger));
    if(!builder){
        std::cout<<" (T_T)~~~, Failed to create builder."<<std::endl;
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));

    if(!network){
        std::cout<<" (T_T)~~~, Failed to create network."<<std::endl;
        return false;
    }

    // 创建 config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config){
        std::cout<<" (T_T)~~~, Failed to create config."<<std::endl;
        return false;
    }

    // 创建parser 从onnx自动构建模型，否则需要自己构建每个算子
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *logger));
    if(!parser){
        std::cout<<" (T_T)~~~, Failed to create parser."<<std::endl;
        return false;
    }

    // 读取onnx模型文件开始构建模型
    auto parsed = parser->parseFromFile(onnx_file_path.c_str(), 1);
    if(!parsed){
        std::cout<<" (T_T)~~~ ,Failed to parse onnx file."<<std::endl;
        return false;
    }


    {
        auto input = network->getInput(0);
        auto input_dims = input->getDimensions();
        auto profile = builder->createOptimizationProfile(); 

        // 配置最小、最优、最大范围
        input_dims.d[0] = 1;                                                         
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = maxbatch;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        config->addOptimizationProfile(profile);

        // 判断是否使用半精度优化模型
        // if(FP16)  
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);

        // 设置默认设备类型为 DLA
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);

        // 获取 DLA 核心支持情况
        int numDLACores = builder->getNbDLACores();
        if (numDLACores > 0) {
            std::cout << "DLA is available. Number of DLA cores: " << numDLACores << std::endl;

            // 设置 DLA 核心
            int coreToUse = 0; // 选择第一个 DLA 核心（可以根据实际需求修改）
            config->setDLACore(coreToUse);
            std::cout << "Using DLA core: " << coreToUse << std::endl;
        } else {
            std::cerr << "DLA not available on this platform, falling back to GPU." << std::endl;
            
            // 如果 DLA 不可用，则设置 GPU 回退
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
        }
    };

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 28);      /*在新的版本中被使用*/

    // 创建序列化引擎文件
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if(!plan){
        std::cout<<" (T_T)~~~, Failed to SerializedNetwork."<<std::endl;
        return false;
    }

    //! 检查输入部分是否符合要求
    auto numInput = network->getNbInputs();
    std::cout<<"模型的输入个数是："<<numInput<<std::endl;
    for(auto i = 0; i<numInput; ++i){
        std::cout<<"    模型的第"<<i<<"个输入：";
        auto mInputDims = network->getInput(i)->getDimensions();
        std::cout<<"  ✨~ model input dims: "<<mInputDims.nbDims <<std::endl;
        for(size_t ii=0; ii<mInputDims.nbDims; ++ii){
            std::cout<<"  ✨^_^ model input dim"<<ii<<": "<<mInputDims.d[ii] <<std::endl;
        }
    }

    auto numOutput = network->getNbOutputs();
    std::cout<<"模型的输出个数是："<<numOutput<<std::endl;
    for(auto i=0; i<numOutput; ++i){
        std::cout<<"    模型的第"<<i<<"个输出：";
        auto mOutputDims = network->getOutput(i)->getDimensions();
            std::cout<<"  ✨~ model output dims: "<<mOutputDims.nbDims <<std::endl;
            for(size_t jj=0; jj<mOutputDims.nbDims; ++jj){
                std::cout<<"  ✨^_^ model output dim"<<jj<<": "<<mOutputDims.d[jj] <<std::endl;
            }
    }

    // 序列化保存推理引擎文件文件
    std::ofstream engine_file(save_engine_path, std::ios::binary);
    if(!engine_file.good()){
        std::cout<<" (T_T)~~~, Failed to open engine file"<<std::endl;
        return false;
    }

    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    std::cout << " ~~Congratulations! 🎉🎉🎉~  Engine build success!!! ✨✨✨~~ " << std::endl;

    return true;

}

} // build_engine