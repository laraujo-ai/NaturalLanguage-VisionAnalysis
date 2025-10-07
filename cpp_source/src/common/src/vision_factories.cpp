#include "../include/factories.hpp"
#include "../../components/stream_handler/include/vision_stream_handlers.hpp"
#include "../../components/frame_sampler/include/frame_samplers.hpp"
#include "../../components/object_detector/include/onnx_object_detector.hpp"
#include <iostream>

namespace vision_analysis {

VisionAnalysisFactory::VisionAnalysisFactory(const VisionAnalysisConfig& config)
    : default_config_(config) {}

std::unique_ptr<IStreamHandler> VisionAnalysisFactory::createStreamHandler(StreamSourceType type) {
    switch (type) {
        case StreamSourceType::RTSP_STREAM:
            return std::make_unique<GStreamerRTSPHandler>();
        case StreamSourceType::VIDEO_FILE:
            return std::make_unique<OpenCVFileHandler>();
        case StreamSourceType::AUTO_DETECT:
        default:
            std::cerr << "Cannot create handler for AUTO_DETECT type. Use createAutoDetectHandler() instead." << std::endl;
            return nullptr;
    }
}

std::unique_ptr<IFrameSampler> VisionAnalysisFactory::createFrameSampler(FrameSamplerType type) {
    switch (type) {
        case FrameSamplerType::UNIFORM:
            return std::make_unique<UniformFrameSampler>();
        case FrameSamplerType::ADAPTIVE:
            return std::make_unique<AdaptiveFrameSampler>();
        case FrameSamplerType::KEYFRAME:
            return std::make_unique<KeyFrameSampler>();
        default:
            std::cerr << "Unknown FrameSampler type" << std::endl;
            return nullptr;
    }
}

std::unique_ptr<IObjectDetector> VisionAnalysisFactory::createObjectDetector(const VisionAnalysisConfig& config) {
    auto detector = std::make_unique<VisionAnalysisDetector>(
        config.yolo_model_path,
        config.clip_model_path,
        config.class_names_file,
        config.num_threads
    );

    detector->setYOLOThresholds(config.confidence_threshold, config.nms_threshold);
    detector->setYOLOInputSize(config.yolo_input_width, config.yolo_input_height);

    return detector;
}

std::unique_ptr<IStorageHandler> VisionAnalysisFactory::createStorageHandler(StorageType type) {
    switch (type) {
        case StorageType::LOCAL_DISK:
            return std::make_unique<LocalDiskStorage>();
        case StorageType::CLOUD_S3:
            return std::make_unique<CloudS3Storage>();
        case StorageType::MILVUS_DB:
            return std::make_unique<MilvusDBStorage>();
        default:
            std::cerr << "Unknown StorageHandler type" << std::endl;
            return nullptr;
    }
}

std::unique_ptr<IStreamHandler> VisionAnalysisFactory::createAutoDetectHandler(const std::string& source) {
    StreamSourceType detected_type = StreamHandlerFactory::detectSourceType(source);
    return createStreamHandler(detected_type);
}

class LocalDiskStorage : public IStorageHandler {
public:
    std::string saveClip(const ClipContainer& clip, const std::string& path) override {
        std::cout << "Saving clip " << clip.clip_id << " to local disk at " << path << std::endl;
        return path + "/" + clip.clip_id + ".mp4";
    }

    bool saveEmbeddings(const std::vector<TrackedObject>& objects) override {
        std::cout << "Saving " << objects.size() << " embeddings to local disk" << std::endl;
        for (const auto& obj : objects) {
            std::cout << "  Object: " << obj.object_id
                     << " (track: " << obj.track_id
                     << ", class: " << obj.class_name
                     << ", embedding_size: " << obj.embedding_history.size() << ")" << std::endl;
        }
        return true;
    }
};

class CloudS3Storage : public IStorageHandler {
public:
    std::string saveClip(const ClipContainer& clip, const std::string& path) override {
        std::cout << "Uploading clip " << clip.clip_id << " to S3 bucket" << std::endl;
        return "s3://" + path + "/" + clip.clip_id + ".mp4";
    }

    bool saveEmbeddings(const std::vector<TrackedObject>& objects) override {
        std::cout << "Uploading " << objects.size() << " embeddings to S3" << std::endl;
        for (const auto& obj : objects) {
            std::cout << "  Object: " << obj.object_id
                     << " (track: " << obj.track_id
                     << ", class: " << obj.class_name
                     << ", embedding_size: " << obj.embedding_history.size() << ")" << std::endl;
        }
        return true;
    }
};

class MilvusDBStorage : public IStorageHandler {
public:
    std::string saveClip(const ClipContainer& clip, const std::string& path) override {
        std::cout << "Storing clip metadata " << clip.clip_id << " in Milvus DB" << std::endl;
        return "milvus://" + clip.clip_id;
    }

    bool saveEmbeddings(const std::vector<TrackedObject>& objects) override {
        std::cout << "Storing " << objects.size() << " embeddings in Milvus DB" << std::endl;
        for (const auto& obj : objects) {
            std::cout << "  Inserting object: " << obj.object_id
                     << " (track: " << obj.track_id
                     << ", class: " << obj.class_name
                     << ", clip: " << obj.clip_id
                     << ", camera: " << obj.camera_id
                     << ", timestamp: " << obj.timestamp
                     << ", embedding_size: " << obj.embedding_history.size()
                     << ", best_frame_embedding_size: " << obj.best_frame_embedding.size() << ")" << std::endl;
        }
        return true;
    }
};

}