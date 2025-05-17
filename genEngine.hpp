#ifndef GENEngine_HPP
#define GENEngine_HPP

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include "TrtLogger.hpp"


namespace build_engine{

    bool genEngine(std::string onnx_file_path, std::string save_engine_path, trtlogger::Logger level, int maxbatch);
}

#endif // TRTMODEL_H

