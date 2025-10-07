#include "../include/factories.hpp"
#include "../../components/stream_handler/include/vision_stream_handlers.hpp"
#include "../../components/frame_sampler/include/frame_samplers.hpp"
#include <iostream>

namespace nl_video_analysis {

std::unique_ptr<IStreamHandler> VisionAnalysisFactory::createStreamHandler(
    StreamSourceType type, const MediaProcessorConfig& config) {

    std::unique_ptr<IStreamHandler> handler;

    switch (type) {
        case StreamSourceType::RTSP_STREAM: {
            auto rtsp_handler = std::make_unique<GStreamerRTSPHandler>(
                config.gst_buffer_size,
                config.queue_max_size,
                config.gst_target_fps,
                config.gst_frame_width,
                config.gst_frame_height
            );
            rtsp_handler->setFramesPerClip(config.frames_per_clip);
            handler = std::move(rtsp_handler);
            break;
        }
        case StreamSourceType::VIDEO_FILE: {
            auto file_handler = std::make_unique<OpenCVFileHandler>(config.frames_per_clip);
            handler = std::move(file_handler);
            break;
        }
        case StreamSourceType::AUTO_DETECT:
        default:
            std::cerr << "Cannot create handler for AUTO_DETECT type. Specify RTSP or FILE." << std::endl;
            return nullptr;
    }

    std::cout << "Created stream handler (type: "
              << (type == StreamSourceType::RTSP_STREAM ? "RTSP" : "FILE") << ")" << std::endl;
    return handler;
}

std::unique_ptr<IFrameSampler> VisionAnalysisFactory::createFrameSampler(FrameSamplerType type) {
    switch (type) {
        case FrameSamplerType::UNIFORM:
            return std::make_unique<UniformFrameSampler>();
        default:
            std::cerr << "Unknown FrameSampler type" << std::endl;
            return nullptr;
    }
}

// Commented out for future use:
//
// std::unique_ptr<IObjectDetector> VisionAnalysisFactory::createObjectDetector(const VisionAnalysisConfig& config) {
//     auto detector = std::make_unique<VisionAnalysisDetector>(
//         config.yolo_model_path,
//         config.clip_model_path,
//         config.class_names_file,
//         config.num_threads
//     );
//     detector->setYOLOThresholds(config.confidence_threshold, config.nms_threshold);
//     detector->setYOLOInputSize(config.yolo_input_width, config.yolo_input_height);
//     return detector;
// }
//
// class LocalDiskStorage : public IStorageHandler {
// public:
//     std::string saveClip(const ClipContainer& clip, const std::string& path) override {
//         std::cout << "Saving clip " << clip.clip_id << " to local disk at " << path << std::endl;
//         return path + "/" + clip.clip_id + ".mp4";
//     }
//     bool saveEmbeddings(const std::vector<TrackedObject>& objects) override {
//         std::cout << "Saving " << objects.size() << " embeddings to local disk" << std::endl;
//         return true;
//     }
// };
//
// std::unique_ptr<IStorageHandler> VisionAnalysisFactory::createStorageHandler(StorageType type) {
//     switch (type) {
//         case StorageType::LOCAL_DISK:
//             return std::make_unique<LocalDiskStorage>();
//         default:
//             std::cerr << "Unknown StorageHandler type" << std::endl;
//             return nullptr;
//     }
// }

}
