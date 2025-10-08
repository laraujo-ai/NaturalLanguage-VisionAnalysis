#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "../../../common/include/interfaces.hpp"
#include "../../../common/include/config_parser.hpp"
#include "../../../common/include/base_model.hpp"
#include "../../stream_handler/include/vision_stream_handlers.hpp"
#include "../../object_detection/include/yolox_detector.hpp"
#include "../../tracker/include/sort_tracker.hpp"
#include "../../frame_sampler/include/frame_samplers.hpp"


namespace nl_video_analysis {

class VideoAnalysisEngine {
private:
    VideoAnalysisConfig config_;

    std::vector<std::unique_ptr<IStreamHandler>> stream_handlers_;
    std::vector<std::string> camera_ids_;

    std::unique_ptr<IFrameSampler> frame_sampler_;
    // clip queue info
    std::queue<ClipContainer> clip_queue_;
    mutable std::mutex clip_queue_mutex_;
    std::condition_variable clip_queue_cv_;
    
    std::vector<std::thread> processing_threads_;
    std::atomic<bool> is_running_;

    void clipProcessingLoop();

    // Commented out for future use:
    std::unique_ptr<YOLOXDetector> object_detector_;
    std::unique_ptr<nl_vision_analysis::SortTracker> tracker_;
    void objectProcessingLoop();

    // std::unique_ptr<IStorageHandler> storage_handler_;
    // std::queue<std::vector<TrackedObject>> tracked_objects_queue_;
    // mutable std::mutex tracked_objects_mutex_;
    // std::condition_variable tracked_objects_cv_;
    // void storageProcessingLoop();

public:
    VideoAnalysisEngine(const VideoAnalysisConfig& config = VideoAnalysisConfig{});
    ~VideoAnalysisEngine();

    bool addSource(const std::string& source_url, const std::string& camera_id, const std::string& source_type);

    void start();
    void stop();

    bool isRunning() const;

    size_t getClipQueueSize() const;
    bool getNextClip(ClipContainer& clip);
    void setConfig(const VideoAnalysisConfig& config);
    VideoAnalysisConfig getConfig() const;
};

}
