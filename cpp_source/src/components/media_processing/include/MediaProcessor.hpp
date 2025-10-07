#pragma once

#include "../../../common/include/interfaces.hpp"
#include "../../../common/include/config_parser.hpp"
#include <vector>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace nl_video_analysis {

class MediaProcessor {
private:
    MediaProcessorConfig config_;

    // Stream handlers for each camera
    std::vector<std::unique_ptr<IStreamHandler>> stream_handlers_;
    std::vector<std::string> camera_ids_;

    // Frame sampler
    std::unique_ptr<IFrameSampler> frame_sampler_;

    // Single queue for clips (with sampled_frames populated)
    std::queue<ClipContainer> clip_queue_;
    mutable std::mutex clip_queue_mutex_;
    std::condition_variable clip_queue_cv_;

    std::vector<std::thread> processing_threads_;
    std::atomic<bool> is_running_;

    // Single processing loop
    void clipProcessingLoop();

    // Commented out for future use:
    // std::unique_ptr<IObjectDetector> object_detector_;
    // std::unique_ptr<IStorageHandler> storage_handler_;
    // std::queue<std::vector<TrackedObject>> tracked_objects_queue_;
    // mutable std::mutex tracked_objects_mutex_;
    // std::condition_variable tracked_objects_cv_;
    // void objectProcessingLoop();
    // void storageProcessingLoop();

public:
    MediaProcessor(const MediaProcessorConfig& config = MediaProcessorConfig{});
    ~MediaProcessor();

    // Add sources from config
    bool addSource(const std::string& source_url, const std::string& camera_id, const std::string& source_type);

    void start();
    void stop();

    bool isRunning() const;

    size_t getClipQueueSize() const;

    // Get next clip with sampled frames
    bool getNextClip(ClipContainer& clip);

    void setConfig(const MediaProcessorConfig& config);
    MediaProcessorConfig getConfig() const;
};

}
