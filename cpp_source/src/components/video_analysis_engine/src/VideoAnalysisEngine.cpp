#include "../include/VideoAnalysisEngine.hpp"

namespace nl_video_analysis {

VideoAnalysisEngine::VideoAnalysisEngine(const VideoAnalysisConfig& config)
    : config_(config), is_running_(false) {
    frame_sampler_ = std::make_unique<UniformFrameSampler>();
    object_detector_ = std::make_unique<YOLOXDetector>(config_.object_detector.weights_path, config_.object_detector.number_of_threads, config_.object_detector.is_fp16, config_.object_detector.classes);
    tracker_ = std::make_unique<nl_video_analysis::SortTracker>(config_.tracker.max_age, config_.tracker.min_hits, config_.tracker.iou_threshold);
    clip_image_encoder_ = std::make_unique<nl_video_analysis::CLIPImageEncoder>(config_.image_encoder.model_path, config_.image_encoder.num_threads, config_.image_encoder.is_fp16); 
    storage_handler_ = std::make_unique<nl_video_analysis::MilvusStorageHandler>(config_.storage_handler.clip_storage_type, 
                                                                                 config_.storage_handler.clip_storage_path,
                                                                                 config_.storage_handler.db_host,
                                                                                 config_.storage_handler.db_port,
                                                                                 config_.storage_handler.db_user,
                                                                                 config_.storage_handler.db_password);
}

VideoAnalysisEngine::~VideoAnalysisEngine() {
    stop();
}

bool VideoAnalysisEngine::addSource(const std::string& source_url,
                                    const std::string& camera_id,
                                    const std::string& source_type,
                                    const StreamCodec& stream_codec) {
    if (stream_handlers_.size() >= static_cast<size_t>(config_.max_connections)) {
        LOG_ERROR("Maximum connections reached ({})", config_.max_connections);
        return false;
    }

    std::unique_ptr<IStreamHandler> handler;

    if (source_type == "rtsp") {
        handler = std::make_unique<GStreamerRTSPHandler>(
            config_.clip_length,
            config_.queue_max_size,
            config_.gst_target_fps,
            config_.gst_frame_width,
            config_.gst_frame_height,
            stream_codec
        );
    } else if (source_type == "file") {
        handler = std::make_unique<OpenCVFileHandler>(config_.clip_length);
    } else {
        LOG_ERROR("Unknown source type: {}", source_type);
        return false;
    }

    std::string final_camera_id = camera_id.empty() ?
        "camera_" + std::to_string(stream_handlers_.size() + 1) : camera_id;

    if (handler->startStream(source_url)) {
        stream_handlers_.push_back(std::move(handler));
        camera_ids_.push_back(final_camera_id);
        LOG_INFO("Camera '{}' added (type: {})", final_camera_id, source_type);
        return true;
    }

    LOG_ERROR("Failed to start stream: {}", source_url);
    return false;
}

void VideoAnalysisEngine::start() {
    if (is_running_) {
        return;
    }

    if (stream_handlers_.empty()) {
        LOG_ERROR("No cameras configured");
        return;
    }

    is_running_ = true;
    clips_processed_ = 0;
    processing_threads_.emplace_back(&VideoAnalysisEngine::clipProcessingLoop, this);
    processing_threads_.emplace_back(&VideoAnalysisEngine::objectProcessingLoop, this);
    processing_threads_.emplace_back(&VideoAnalysisEngine::benchmarkReportingLoop, this);

    LOG_INFO("Pipeline started ({} camera(s))", stream_handlers_.size());
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

    // Log final benchmark report
    std::string final_report = PipelineBenchmark::getInstance().generateReport();
    LOG_INFO("=== Final Benchmark Report (Total Clips: {}) ==={}", clips_processed_.load(), final_report);

    LOG_INFO("Pipeline stopped");
}

bool VideoAnalysisEngine::isRunning() const {
    return is_running_;
}

void VideoAnalysisEngine::clipProcessingLoop() {
    while (is_running_) {
        for (size_t i = 0; i < stream_handlers_.size(); ++i) {
            auto& handler = stream_handlers_[i];
            if (handler->isActive()) {
                std::optional<ClipContainer> clip;

                // Benchmark clip retrieval (includes network/file I/O latency)
                {
                    ScopedTimer timer("clip_retrieval", camera_ids_[i]);
                    clip = handler->getNextClip();
                }

                if (clip.has_value()) {
                    clip.value().camera_id = camera_ids_[i];

                    // Benchmark frame sampling (actual processing only)
                    {
                        ScopedTimer timer("frame_sampling", camera_ids_[i]);
                        frame_sampler_->sampleFrames(clip.value(), config_.sampled_frames_count);
                    }

                    std::unique_lock<std::mutex> lock(clip_queue_mutex_);
                    if (clip_queue_.size() < static_cast<size_t>(config_.queue_max_size)) {
                        clip_queue_.push(std::move(clip.value()));
                        clip_queue_cv_.notify_one();
                    } else {
                        LOG_WARN("Queue full, dropping clip from camera '{}'", camera_ids_[i]);
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void VideoAnalysisEngine::objectProcessingLoop() {
    while (is_running_) {
        ClipContainer clip;
        if (!getNextClip(clip)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ScopedTimer clip_timer("clip_total_processing", clip.camera_id);
        std::vector<std::vector<Detection>> all_detections;
        {
            ScopedTimer detection_timer("clip_object_detection", clip.camera_id);
            all_detections.reserve(clip.sampled_frames.size());

            for (const auto& frame : clip.sampled_frames) {
                std::vector<Detection> detections = object_detector_->detect(
                    frame,
                    config_.object_detector.conf_threshold,
                    config_.object_detector.nms_threshold
                );
                all_detections.push_back(std::move(detections));
            }
        }

        std::vector<std::vector<nlohmann::json>> all_tracked_objects;
        all_tracked_objects.reserve(all_detections.size());

        for (const auto& detections : all_detections) {
            std::vector<nlohmann::json> tracked_objects = tracker_->track(detections);
            all_tracked_objects.push_back(std::move(tracked_objects));
        }
        std::string base_path = "/home/nvidia/projects/NaturalLanguage-VisionAnalysis/test_cropps/";
        std::map<int64_t, std::vector<std::vector<float>>> tracklet_to_embeddings;
        for (size_t i = 0; i < clip.sampled_frames.size(); ++i) {
            const auto& frame = clip.sampled_frames[i];
            const auto& tracked_objects = all_tracked_objects[i];

            for (const auto& tracklet : tracked_objects) {
                auto bbox = tracklet["BoundingBox"];
                int64_t tracker_id = tracklet["TrackerId"].get<int64_t>();

                std::optional<cv::Mat> cropped = crop_object(
                    frame,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    10
                );
                if (cropped) {
                    std::vector<float> embedding = clip_image_encoder_->encode(cropped.value());
                    tracklet_to_embeddings[tracker_id].push_back(embedding);
                }
            }
        }
        storage_handler_->saveClip(clip, tracklet_to_embeddings);
        clips_processed_++;
    }
}

void VideoAnalysisEngine::benchmarkReportingLoop() {
    const int report_interval_seconds = 30;

    while (is_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(report_interval_seconds));

        if (!is_running_) break;

        // Generate and log benchmark report
        std::string report = PipelineBenchmark::getInstance().generateReport();
        LOG_INFO("=== Benchmark Report (Clips Processed: {}) ==={}", clips_processed_.load(), report);
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
