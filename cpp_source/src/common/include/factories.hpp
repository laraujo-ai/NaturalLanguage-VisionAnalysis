#pragma once

#include "interfaces.hpp"
#include "config_parser.hpp"
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

// Commented out for future use:
// enum class StorageType {
//     LOCAL_DISK,
//     CLOUD_S3,
//     MILVUS_DB
// };

class IComponentFactory {
public:
    virtual ~IComponentFactory() = default;
    virtual std::unique_ptr<IStreamHandler> createStreamHandler(StreamSourceType type, const MediaProcessorConfig& config) = 0;
    virtual std::unique_ptr<IFrameSampler> createFrameSampler(FrameSamplerType type) = 0;
    // Commented out for future use:
    // virtual std::unique_ptr<IObjectDetector> createObjectDetector(const VisionAnalysisConfig& config) = 0;
    // virtual std::unique_ptr<IStorageHandler> createStorageHandler(StorageType type) = 0;
};

class VisionAnalysisFactory : public IComponentFactory {
public:
    VisionAnalysisFactory() = default;

    std::unique_ptr<IStreamHandler> createStreamHandler(StreamSourceType type, const MediaProcessorConfig& config) override;
    std::unique_ptr<IFrameSampler> createFrameSampler(FrameSamplerType type) override;

    // Commented out for future use:
    // std::unique_ptr<IObjectDetector> createObjectDetector(const VisionAnalysisConfig& config);
    // std::unique_ptr<IStorageHandler> createStorageHandler(StorageType type);
};

}