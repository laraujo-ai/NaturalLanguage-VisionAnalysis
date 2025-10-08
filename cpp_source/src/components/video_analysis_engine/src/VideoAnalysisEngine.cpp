#include "../include/VideoAnalysisEngine.hpp"

namespace nl_video_analysis {

VideoAnalysisEngine::VideoAnalysisEngine(const VideoAnalysisConfig& config)
    : config_(config), is_running_(false) {

    frame_sampler_ = std::make_unique<UniformFrameSampler>();
    object_detector_ = std::make_unique<YOLOXDetector>(config_.object_detector.weights_path, config_.object_detector.number_of_threads);
    tracker_ = std::make_unique<nl_vision_analysis::SortTracker>(config_.tracker.max_age, config_.tracker.min_hits, config_.tracker.iou_threshold);
}

VideoAnalysisEngine::~VideoAnalysisEngine() {
    stop();
}

bool VideoAnalysisEngine::addSource(const std::string& source_url, const std::string& camera_id, const std::string& source_type) {
    if (stream_handlers_.size() >= static_cast<size_t>(config_.max_connections)) {
        std::cerr << "Maximum connections reached (" << config_.max_connections << ")" << std::endl;
        return false;
    }

    std::unique_ptr<IStreamHandler> handler;

    // Create stream handler directly based on type
    if (source_type == "rtsp") {
        auto rtsp_handler = std::make_unique<GStreamerRTSPHandler>(
            config_.clip_length,
            config_.queue_max_size,
            config_.gst_target_fps,
            config_.gst_frame_width,
            config_.gst_frame_height
        );
        handler = std::move(rtsp_handler);
    } else if (source_type == "file") {
        handler = std::make_unique<OpenCVFileHandler>(config_.clip_length);
    } else {
        std::cerr << "Unknown source type: " << source_type << std::endl;
        return false;
    }

    std::string final_camera_id = camera_id.empty() ?
        "camera_" + std::to_string(stream_handlers_.size() + 1) : camera_id;

    if (handler->startStream(source_url)) {
        stream_handlers_.push_back(std::move(handler));
        camera_ids_.push_back(final_camera_id);
        std::cout << "Added source: " << source_url << " as camera: " << final_camera_id
                  << " (type: " << source_type << ")" << std::endl;
        return true;
    }

    std::cerr << "Failed to start stream: " << source_url << std::endl;
    return false;
}

void VideoAnalysisEngine::start() {
    if (is_running_) {
        std::cout << "VideoAnalysisEngine already running" << std::endl;
        return;
    }

    if (stream_handlers_.empty()) {
        std::cerr << "No stream handlers added. Cannot start." << std::endl;
        return;
    }

    is_running_ = true;

    processing_threads_.emplace_back(&VideoAnalysisEngine::clipProcessingLoop, this);
    processing_threads_.emplace_back(&VideoAnalysisEngine::objectProcessingLoop, this);

    std::cout << "VideoAnalysisEngine started with " << stream_handlers_.size()
              << " streams and " << processing_threads_.size() << " processing thread(s)" << std::endl;
}

void VideoAnalysisEngine::stop() {
    if (!is_running_) {
        return;
    }
    is_running_ = false;

    for (auto& handler : stream_handlers_) {
        handler->stopStream();
    }

    clip_queue_cv_.notify_all();

    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    processing_threads_.clear();
    stream_handlers_.clear();
    camera_ids_.clear();

    std::cout << "VideoAnalysisEngine stopped" << std::endl;
}

bool VideoAnalysisEngine::isRunning() const {
    return is_running_;
}

void VideoAnalysisEngine::clipProcessingLoop() {
    while (is_running_) {
        for (size_t i = 0; i < stream_handlers_.size(); ++i) {
            auto& handler = stream_handlers_[i];
            if (handler->isActive()) {
                auto clip = handler->getNextClip();
                if (clip.has_value()) {
                    // Set camera_id from our stored list. The id here will probrably be a unique name for the camera so a string.
                    clip.value().camera_id = camera_ids_[i];

                    frame_sampler_->sampleFrames(clip.value(), config_.sampled_frames_count);

                    std::unique_lock<std::mutex> lock(clip_queue_mutex_);
                    if (clip_queue_.size() < static_cast<size_t>(config_.queue_max_size)) {
                        std::cout << "Clip processed from camera: " << camera_ids_[i]
                                  << " (frames: " << clip.value().frames.size()
                                  << ", sampled: " << clip.value().sampled_frames.size()
                                  << ", queue size: " << clip_queue_.size() + 1 << ")" << std::endl;

                        clip_queue_.push(std::move(clip.value()));
                        clip_queue_cv_.notify_one();
                    } else {
                        std::cerr << "Clip queue full, dropping clip from " << camera_ids_[i] << std::endl;
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void VideoAnalysisEngine::objectProcessingLoop()
{
    while(is_running_)
    {
        ClipContainer clip;
        this->getNextClip(clip);
        for(auto frame : clip.sampled_frames)
        {
            std::vector<Detection> detectionResult = object_detector_->detect(frame, this->config_.object_detector.conf_threshold, this->config_.object_detector.nms_threshold);
            std::vector<nlohmann::json> trackedObjects = tracker_->track(detectionResult);
            
            for (int i(0); i < detectionResult.size(); i++)
            {
                std::cout << "Detection  :" << i << std::endl;
                std::cout << "confidence :" << detectionResult[i].score << std::endl;
            }
        }    
    }
}

size_t VideoAnalysisEngine::getClipQueueSize() const {
    std::lock_guard<std::mutex> lock(clip_queue_mutex_);
    return clip_queue_.size();
}

bool VideoAnalysisEngine::getNextClip(ClipContainer& clip) {
    std::lock_guard<std::mutex> lock(clip_queue_mutex_);
    if (clip_queue_.empty()) {
        return false;
    }
    clip = std::move(clip_queue_.front());
    clip_queue_.pop();
    return true;
}

void VideoAnalysisEngine::setConfig(const VideoAnalysisConfig& config) {
    config_ = config;
}

VideoAnalysisConfig VideoAnalysisEngine::getConfig() const {
    return config_;
}

}
