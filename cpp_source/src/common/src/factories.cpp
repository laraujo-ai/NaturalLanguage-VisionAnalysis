#include "../include/factories.hpp"
#include "../../../components/stream_handler/include/stream_handlers.hpp"
#include "../../../components/frame_sampler/include/frame_samplers.hpp"
#include "../../../components/object_detector/include/object_detectors.hpp"
#include <iostream>

namespace vision_analysis {

std::unique_ptr<IStreamHandler> ComponentFactory::createStreamHandler(StreamHandlerType type) {
    return StreamHandlerFactory::create(type);
}

std::unique_ptr<IFrameSampler> ComponentFactory::createFrameSampler(FrameSamplerType type) {
    return FrameSamplerFactory::create(type);
}

std::unique_ptr<IObjectDetector> ComponentFactory::createObjectDetector(ObjectDetectorType type) {
    return ObjectDetectorFactory::create(type);
}

std::unique_ptr<IStorageHandler> ComponentFactory::createStorageHandler(StorageType type) {
    return StorageHandlerFactory::create(type);
}

std::unique_ptr<IStreamHandler> StreamHandlerFactory::create(StreamHandlerType type, const std::string& config) {
    switch (type) {
        case StreamHandlerType::RTSP:
            return std::make_unique<RTSPStreamHandler>();
        case StreamHandlerType::FILE:
            return std::make_unique<FileStreamHandler>();
        case StreamHandlerType::WEBCAM:
            return std::make_unique<WebcamStreamHandler>();
        default:
            std::cerr << "Unknown StreamHandler type" << std::endl;
            return nullptr;
    }
}

std::unique_ptr<IFrameSampler> FrameSamplerFactory::create(FrameSamplerType type) {
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

std::unique_ptr<IObjectDetector> ObjectDetectorFactory::create(ObjectDetectorType type, const std::string& model_path) {
    std::string path = model_path.empty() ? "default_model.onnx" : model_path;

    switch (type) {
        case ObjectDetectorType::YOLO:
            return std::make_unique<YOLODetector>(path);
        case ObjectDetectorType::SSD:
            return std::make_unique<SSDDetector>(path);
        case ObjectDetectorType::FASTER_RCNN:
            return std::make_unique<FasterRCNNDetector>(path);
        default:
            std::cerr << "Unknown ObjectDetector type" << std::endl;
            return nullptr;
    }
}

class LocalDiskStorage : public IStorageHandler {
public:
    std::string saveClip(const ClipContainer& clip, const std::string& path) override {
        std::cout << "Saving clip " << clip.clip_id << " to local disk at " << path << std::endl;
        return path + "/" + clip.clip_id + ".mp4";
    }

    bool saveEmbeddings(const std::vector<TrackedObject>& objects) override {
        std::cout << "Saving " << objects.size() << " embeddings to local disk" << std::endl;
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
        return true;
    }
};

std::unique_ptr<IStorageHandler> StorageHandlerFactory::create(StorageType type, const std::string& config) {
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

}