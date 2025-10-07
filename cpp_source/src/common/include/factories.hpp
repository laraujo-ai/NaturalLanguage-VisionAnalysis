#pragma once

#include "interfaces.hpp"
#include <memory>
#include <string>

namespace nl_video_analysis {

enum class StreamSourceType {
    RTSP_STREAM,
    VIDEO_FILE,
    AUTO_DETECT
};

enum class FrameSamplerType {
    UNIFORM,
};

enum class StorageType {
    LOCAL_DISK,
};

struct VisionAnalysisConfig {
    std::string yolo_model_path = "models/yolo.onnx";
    std::string clip_model_path = "models/clip.onnx";
    std::string class_names_file = "models/coco.names";
    int num_threads = 4;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int yolo_input_width = 640;
    int yolo_input_height = 640;
    int clip_input_size = 224;
    int clip_embedding_size = 512;
};

class IComponentFactory {
public:
    virtual ~IComponentFactory() = default;
    virtual std::unique_ptr<IStreamHandler> createStreamHandler(StreamSourceType type) = 0;
    virtual std::unique_ptr<IFrameSampler> createFrameSampler(FrameSamplerType type) = 0;
    virtual std::unique_ptr<IObjectDetector> createObjectDetector(const VisionAnalysisConfig& config) = 0;
    virtual std::unique_ptr<IStorageHandler> createStorageHandler(StorageType type) = 0;
};

class VisionAnalysisFactory : public IComponentFactory {
private:
    VisionAnalysisConfig default_config_;

public:
    VisionAnalysisFactory(const VisionAnalysisConfig& config = VisionAnalysisConfig{});

    std::unique_ptr<IStreamHandler> createStreamHandler(StreamSourceType type) override;
    std::unique_ptr<IStreamHandler> createAutoDetectHandler(const std::string& source);
    std::unique_ptr<IFrameSampler> createFrameSampler(FrameSamplerType type) override;
    std::unique_ptr<IObjectDetector> createObjectDetector(const VisionAnalysisConfig& config) override;
    std::unique_ptr<IStorageHandler> createStorageHandler(StorageType type) override;

    void setDefaultConfig(const VisionAnalysisConfig& config) { default_config_ = config; }
    VisionAnalysisConfig getDefaultConfig() const { return default_config_; }
};

}