#include "../include/MediaProcessor.hpp"
#include <iostream>
#include <algorithm>

namespace vision_analysis {

MediaProcessor::MediaProcessor(const MediaProcessorConfig& config)
    : config_(config), is_running_(false) {
    factory_ = std::make_unique<VisionAnalysisFactory>(config_.vision_config);

    frame_sampler_ = factory_->createFrameSampler(config_.sampler_type);
    object_detector_ = factory_->createObjectDetector(config_.vision_config);
    storage_handler_ = factory_->createStorageHandler(config_.storage_type);

    std::cout << "MediaProcessor created with ONNX-based vision analysis pipeline" << std::endl;
}

MediaProcessor::~MediaProcessor() {
    stop();
}

bool MediaProcessor::addSource(const std::string& source_url, const std::string& camera_id) {
    if (stream_handlers_.size() >= config_.max_connections) {
        std::cerr << "Maximum connections reached" << std::endl;
        return false;
    }

    auto handler = factory_->createAutoDetectHandler(source_url);
    if (!handler) {
        std::cerr << "Failed to create handler for source: " << source_url << std::endl;
        return false;
    }

    std::string final_camera_id = camera_id.empty() ?
        "camera_" + std::to_string(stream_handlers_.size() + 1) : camera_id;

    if (handler->startStream(source_url)) {
        stream_handlers_.push_back(std::move(handler));
        std::cout << "Added source: " << source_url << " for camera: " << final_camera_id << std::endl;
        return true;
    }

    return false;
}

bool MediaProcessor::addRTSPConnection(const std::string& rtsp_url, const std::string& camera_id) {
    if (stream_handlers_.size() >= config_.max_connections) {
        std::cerr << "Maximum connections reached" << std::endl;
        return false;
    }

    auto handler = factory_->createStreamHandler(StreamSourceType::RTSP_STREAM);
    if (!handler) {
        std::cerr << "Failed to create RTSP handler" << std::endl;
        return false;
    }

    std::string final_camera_id = camera_id.empty() ?
        "rtsp_camera_" + std::to_string(stream_handlers_.size() + 1) : camera_id;

    if (handler->startStream(rtsp_url)) {
        stream_handlers_.push_back(std::move(handler));
        std::cout << "Added RTSP connection: " << rtsp_url << " for camera: " << final_camera_id << std::endl;
        return true;
    }

    return false;
}

bool MediaProcessor::addVideoFile(const std::string& file_path, const std::string& camera_id) {
    if (stream_handlers_.size() >= config_.max_connections) {
        std::cerr << "Maximum connections reached" << std::endl;
        return false;
    }

    auto handler = factory_->createStreamHandler(StreamSourceType::VIDEO_FILE);
    if (!handler) {
        std::cerr << "Failed to create file handler" << std::endl;
        return false;
    }

    std::string final_camera_id = camera_id.empty() ?
        "file_" + std::to_string(stream_handlers_.size() + 1) : camera_id;

    if (handler->startStream(file_path)) {
        stream_handlers_.push_back(std::move(handler));
        std::cout << "Added video file: " << file_path << " for camera: " << final_camera_id << std::endl;
        return true;
    }

    return false;
}

void MediaProcessor::start() {
    if (is_running_) {
        std::cout << "MediaProcessor already running" << std::endl;
        return;
    }

    is_running_ = true;

    processing_threads_.emplace_back(&MediaProcessor::clipProcessingLoop, this);
    processing_threads_.emplace_back(&MediaProcessor::frameProcessingLoop, this);
    processing_threads_.emplace_back(&MediaProcessor::objectProcessingLoop, this);
    processing_threads_.emplace_back(&MediaProcessor::storageProcessingLoop, this);

    std::cout << "MediaProcessor started with " << processing_threads_.size() << " threads" << std::endl;
}

void MediaProcessor::stop() {
    if (!is_running_) {
        return;
    }

    is_running_ = false;

    for (auto& handler : stream_handlers_) {
        handler->stopStream();
    }

    clip_queue_cv_.notify_all();
    sampled_frames_cv_.notify_all();
    tracked_objects_cv_.notify_all();

    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    processing_threads_.clear();
    stream_handlers_.clear();

    std::cout << "MediaProcessor stopped" << std::endl;
}

bool MediaProcessor::isRunning() const {
    return is_running_;
}

void MediaProcessor::clipProcessingLoop() {
    while (is_running_) {
        for (auto& handler : stream_handlers_) {
            if (handler->isActive()) {
                auto clip = handler->getNextClip();
                if (clip.has_value()) {
                    std::unique_lock<std::mutex> lock(clip_queue_mutex_);
                    if (clip_queue_.size() < config_.queue_max_size) {
                        clip_queue_.push(std::move(clip.value()));
                        clip_queue_cv_.notify_one();
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void MediaProcessor::frameProcessingLoop() {
    while (is_running_) {
        std::unique_lock<std::mutex> lock(clip_queue_mutex_);
        clip_queue_cv_.wait(lock, [this] { return !clip_queue_.empty() || !is_running_; });

        if (!clip_queue_.empty()) {
            ClipContainer clip = std::move(clip_queue_.front());
            clip_queue_.pop();
            lock.unlock();

            SampledFrames sampled = frame_sampler_->sampleFrames(clip, config_.sampled_frames_count);

            std::unique_lock<std::mutex> sampled_lock(sampled_frames_mutex_);
            if (sampled_frames_queue_.size() < config_.queue_max_size) {
                sampled_frames_queue_.push(std::move(sampled));
                sampled_frames_cv_.notify_one();
            }
        }
    }
}

void MediaProcessor::objectProcessingLoop() {
    while (is_running_) {
        std::unique_lock<std::mutex> lock(sampled_frames_mutex_);
        sampled_frames_cv_.wait(lock, [this] { return !sampled_frames_queue_.empty() || !is_running_; });

        if (!sampled_frames_queue_.empty()) {
            SampledFrames frames = std::move(sampled_frames_queue_.front());
            sampled_frames_queue_.pop();
            lock.unlock();

            std::vector<TrackedObject> tracked = object_detector_->detectAndTrack(frames);

            std::unique_lock<std::mutex> tracked_lock(tracked_objects_mutex_);
            if (tracked_objects_queue_.size() < config_.queue_max_size) {
                tracked_objects_queue_.push(std::move(tracked));
                tracked_objects_cv_.notify_one();
            }
        }
    }
}

void MediaProcessor::storageProcessingLoop() {
    while (is_running_) {
        std::unique_lock<std::mutex> lock(tracked_objects_mutex_);
        tracked_objects_cv_.wait(lock, [this] { return !tracked_objects_queue_.empty() || !is_running_; });

        if (!tracked_objects_queue_.empty()) {
            std::vector<TrackedObject> objects = std::move(tracked_objects_queue_.front());
            tracked_objects_queue_.pop();
            lock.unlock();

            storage_handler_->saveEmbeddings(objects);
        }
    }
}

size_t MediaProcessor::getClipQueueSize() const {
    std::lock_guard<std::mutex> lock(clip_queue_mutex_);
    return clip_queue_.size();
}

size_t MediaProcessor::getSampledFramesQueueSize() const {
    std::lock_guard<std::mutex> lock(sampled_frames_mutex_);
    return sampled_frames_queue_.size();
}

size_t MediaProcessor::getTrackedObjectsQueueSize() const {
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);
    return tracked_objects_queue_.size();
}

void MediaProcessor::setConfig(const MediaProcessorConfig& config) {
    config_ = config;
}

MediaProcessorConfig MediaProcessor::getConfig() const {
    return config_;
}

}