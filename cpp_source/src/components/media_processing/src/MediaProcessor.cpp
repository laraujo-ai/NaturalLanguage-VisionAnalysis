#include "../include/MediaProcessor.hpp"
#include <iostream>
#include <algorithm>

namespace nl_video_analysis {

MediaProcessor::MediaProcessor(const MediaProcessorConfig& config)
    : config_(config), is_running_(false) {

    // Create factory
    factory_ = std::make_unique<VisionAnalysisFactory>();

    // Create frame sampler based on config
    FrameSamplerType sampler_type = FrameSamplerType::UNIFORM;
    if (config_.sampler_type == "uniform") {
        sampler_type = FrameSamplerType::UNIFORM;
    }
    // Add more sampler types as needed

    frame_sampler_ = factory_->createFrameSampler(sampler_type);

    std::cout << "MediaProcessor created - Stream handling + Frame sampling only" << std::endl;
    std::cout << "  Max connections: " << config_.max_connections << std::endl;
    std::cout << "  Frames per clip: " << config_.frames_per_clip << std::endl;
    std::cout << "  Sampled frames: " << config_.sampled_frames_count << std::endl;
    std::cout << "  Sampler type: " << config_.sampler_type << std::endl;
}

MediaProcessor::~MediaProcessor() {
    stop();
}

bool MediaProcessor::addSource(const std::string& source_url, const std::string& camera_id, const std::string& source_type) {
    if (stream_handlers_.size() >= static_cast<size_t>(config_.max_connections)) {
        std::cerr << "Maximum connections reached (" << config_.max_connections << ")" << std::endl;
        return false;
    }

    StreamSourceType type = StreamSourceType::AUTO_DETECT;
    if (source_type == "rtsp") {
        type = StreamSourceType::RTSP_STREAM;
    } else if (source_type == "file") {
        type = StreamSourceType::VIDEO_FILE;
    }

    auto handler = factory_->createStreamHandler(type, config_);
    if (!handler) {
        std::cerr << "Failed to create handler for source: " << source_url << std::endl;
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

void MediaProcessor::start() {
    if (is_running_) {
        std::cout << "MediaProcessor already running" << std::endl;
        return;
    }

    if (stream_handlers_.empty()) {
        std::cerr << "No stream handlers added. Cannot start." << std::endl;
        return;
    }

    is_running_ = true;

    // Start single clip processing thread
    processing_threads_.emplace_back(&MediaProcessor::clipProcessingLoop, this);

    std::cout << "MediaProcessor started with " << stream_handlers_.size()
              << " streams and " << processing_threads_.size() << " processing thread(s)" << std::endl;
}

void MediaProcessor::stop() {
    if (!is_running_) {
        return;
    }

    std::cout << "Stopping MediaProcessor..." << std::endl;
    is_running_ = false;

    // Stop all streams
    for (auto& handler : stream_handlers_) {
        handler->stopStream();
    }

    // Notify all waiting threads
    clip_queue_cv_.notify_all();

    // Join all threads
    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    processing_threads_.clear();
    stream_handlers_.clear();
    camera_ids_.clear();

    std::cout << "MediaProcessor stopped" << std::endl;
}

bool MediaProcessor::isRunning() const {
    return is_running_;
}

void MediaProcessor::clipProcessingLoop() {
    std::cout << "Clip processing thread started" << std::endl;

    while (is_running_) {
        for (size_t i = 0; i < stream_handlers_.size(); ++i) {
            auto& handler = stream_handlers_[i];
            if (handler->isActive()) {
                auto clip = handler->getNextClip();
                if (clip.has_value()) {
                    // Set camera_id from our stored list
                    clip.value().camera_id = camera_ids_[i];

                    // Sample frames directly on the clip
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

    std::cout << "Clip processing thread stopped" << std::endl;
}

size_t MediaProcessor::getClipQueueSize() const {
    std::lock_guard<std::mutex> lock(clip_queue_mutex_);
    return clip_queue_.size();
}

bool MediaProcessor::getNextClip(ClipContainer& clip) {
    std::lock_guard<std::mutex> lock(clip_queue_mutex_);
    if (clip_queue_.empty()) {
        return false;
    }
    clip = std::move(clip_queue_.front());
    clip_queue_.pop();
    return true;
}

void MediaProcessor::setConfig(const MediaProcessorConfig& config) {
    config_ = config;
}

MediaProcessorConfig MediaProcessor::getConfig() const {
    return config_;
}

}
