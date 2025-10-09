#ifndef ONNX_SESSION_HPP
#define ONNX_SESSION_HPP

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "logger.hpp"
#include <string>
#include <memory>
#include <iostream>

class ONNXSessionBuilder {
public:
    ONNXSessionBuilder(const std::string& model_path, int num_threads);
    ~ONNXSessionBuilder() = default;

    std::unique_ptr<Ort::Session> build();
    static Ort::Env& getEnv();

private:
    std::string model_path_;
    int num_threads_;

    OrtTensorRTProviderOptions getTensorRTOptions();
    OrtCUDAProviderOptions getCUDAOptions();
};

inline Ort::Env& ONNXSessionBuilder::getEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXSession");
    return env;
}

inline ONNXSessionBuilder::ONNXSessionBuilder(const std::string& model_path, int num_threads)
    : model_path_(model_path)
    , num_threads_(num_threads)
{
}

inline OrtTensorRTProviderOptions ONNXSessionBuilder::getTensorRTOptions() {
    OrtTensorRTProviderOptions trtOptions{};
    trtOptions.device_id = 0;
    trtOptions.trt_max_workspace_size = 2147483648;
    trtOptions.trt_fp16_enable = 1;
    trtOptions.trt_engine_cache_enable = 1;
    trtOptions.trt_engine_cache_path = "./trt_cache";
    return trtOptions;
}

inline OrtCUDAProviderOptions ONNXSessionBuilder::getCUDAOptions() {
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    cudaOptions.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
    cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cudaOptions.do_copy_in_default_stream = 1;
    return cudaOptions;
}

inline std::unique_ptr<Ort::Session> ONNXSessionBuilder::build() {
    Ort::SessionOptions sessionOptions;
    Ort::Env& env = getEnv();  
    sessionOptions.SetIntraOpNumThreads(num_threads_);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtTensorRTProviderOptions trtOptions = getTensorRTOptions();
        sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
        LOG_INFO("TensorRT execution provider enabled");
    } catch (const Ort::Exception& e) {
        LOG_DEBUG("TensorRT not available");
    }

    try {
        OrtCUDAProviderOptions cudaOptions = getCUDAOptions();
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        LOG_INFO("CUDA execution provider enabled");
    } catch (const Ort::Exception& e) {
        LOG_DEBUG("CUDA not available");
    }

    LOG_INFO("Loading ONNX model: {}", model_path_);

    try {
        return std::make_unique<Ort::Session>(getEnv(), model_path_.c_str(), sessionOptions);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to create ONNX session: {}", e.what());
        throw;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create ONNX session: {}", e.what());
        throw;
    }
}

#endif // ONNX_SESSION_HPP