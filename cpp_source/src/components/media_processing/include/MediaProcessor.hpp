#pragma once

#include "../../../common/include/interfaces.hpp"
#include "../../../common/include/factories.hpp"
#include <vector>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace vision_analysis {

struct MediaProcessorConfig {
    VisionAnalysisConfig vision_config;
    FrameSamplerType sampler_type = FrameSamplerType::UNIFORM;
    StorageType storage_type = StorageType::LOCAL_DISK;

    int frames_per_clip = 30;
    int sampled_frames_count = 5;
    int max_connections = 10;
    int queue_max_size = 100;
    std::string storage_path = "./storage";
};

class MediaProcessor {
private:
    MediaProcessorConfig config_;
    std::unique_ptr<VisionAnalysisFactory> factory_;

    std::vector<std::unique_ptr<IStreamHandler>> stream_handlers_;
    std::unique_ptr<IFrameSampler> frame_sampler_;
    std::unique_ptr<IObjectDetector> object_detector_;
    std::unique_ptr<IStorageHandler> storage_handler_;

    std::queue<ClipContainer> clip_queue_;
    std::queue<SampledFrames> sampled_frames_queue_;
    std::queue<std::vector<TrackedObject>> tracked_objects_queue_;

    mutable std::mutex clip_queue_mutex_;
    mutable std::mutex sampled_frames_mutex_;
    mutable std::mutex tracked_objects_mutex_;

    std::condition_variable clip_queue_cv_;
    std::condition_variable sampled_frames_cv_;
    std::condition_variable tracked_objects_cv_;

    std::vector<std::thread> processing_threads_;
    std::atomic<bool> is_running_;

    void clipProcessingLoop();
    void frameProcessingLoop();
    void objectProcessingLoop();
    void storageProcessingLoop();

public:
    MediaProcessor(const MediaProcessorConfig& config = MediaProcessorConfig{});
    ~MediaProcessor();

    bool addSource(const std::string& source_url, const std::string& camera_id = "");
    bool addRTSPConnection(const std::string& rtsp_url, const std::string& camera_id = "");
    bool addVideoFile(const std::string& file_path, const std::string& camera_id = "");

    void start();
    void stop();

    bool isRunning() const;

    size_t getClipQueueSize() const;
    size_t getSampledFramesQueueSize() const;
    size_t getTrackedObjectsQueueSize() const;

    void setConfig(const MediaProcessorConfig& config);
    MediaProcessorConfig getConfig() const;
};

}