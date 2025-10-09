#include "../include/vision_stream_handlers.hpp"
#include "../../../common/include/logger.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <algorithm>

namespace nl_video_analysis {

GStreamerRTSPHandler::GStreamerRTSPHandler(int clip_length, int max_queue_size,
                                           int target_fps, int target_width, int target_height,
                                           StreamCodec codec)
    : is_active_(false), max_queue_size_(max_queue_size), clip_length_(clip_length),
      target_fps_(target_fps), target_width_(target_width), target_height_(target_height),
      stream_codec_(codec), pipeline_(nullptr), appsink_(nullptr), bus_(nullptr), main_loop_(nullptr),
      stream_start_pts_ms_(0) {

    gst_init(nullptr, nullptr);
    clip_start_time_ = std::chrono::steady_clock::now();
    this->frames_per_clip_ = this->target_fps_ * this->clip_length_;
}

GStreamerRTSPHandler::~GStreamerRTSPHandler() {
    stopStream();
}

bool GStreamerRTSPHandler::startStream(const std::string& rtsp_url) {
    if (is_active_) {
        return false;
    }

    rtsp_url_ = rtsp_url;

    if (camera_id_.empty()) {
        camera_id_ = "rtsp_camera_" + std::to_string(std::hash<std::string>{}(rtsp_url) % 10000);
    }

    if (!initializeGStreamer()) {
        LOG_ERROR("GStreamer initialization failed");
        return false;
    }

    is_active_ = true;
    gst_thread_ = std::thread(&GStreamerRTSPHandler::gstreamerLoop, this);
    capture_thread_ = std::thread(&GStreamerRTSPHandler::captureLoop, this);

    return true;
}

void GStreamerRTSPHandler::stopStream() {
    if (!is_active_) return;

    is_active_ = false;

    if (main_loop_ && g_main_loop_is_running(main_loop_)) {
        g_main_loop_quit(main_loop_);
    }

    if (gst_thread_.joinable()) {
        gst_thread_.join();
    }

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    cleanupGStreamer();

    queue_cv_.notify_all();
}

std::optional<ClipContainer> GStreamerRTSPHandler::getNextClip() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return !clip_queue_.empty() || !is_active_; });

    if (clip_queue_.empty()) {
        return std::nullopt;
    }

    ClipContainer clip = std::move(clip_queue_.front());
    clip_queue_.pop();
    return clip;
}

bool GStreamerRTSPHandler::isActive() const {
    return is_active_;
}

std::string GStreamerRTSPHandler::buildNvidiaHardwarePipeline() const {
    std::stringstream pipeline_str;

    // NVIDIA hardware-accelerated pipeline based on the Python implementation for the action recognition pipeline
    pipeline_str << "rtspsrc location=\"" << rtsp_url_ << "\" latency=50 protocol=tcp ! "
                 << getDepayElement() << " ! "
                 << getParserElement() << " ! "
                 << "nvv4l2decoder enable-max-performance=1 ! "
                 << "nvvideoconvert ! "
                 << "videorate ! "
                 << "video/x-raw,width=" << target_width_ << ",height=" << target_height_
                 << ",framerate=" << target_fps_ << "/1 ! "
                 << "videoconvert ! "
                 << "video/x-raw,format=BGR ! "
                 << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";

    return pipeline_str.str();
}

std::string GStreamerRTSPHandler::getDepayElement() const {
    switch (stream_codec_) {
        case StreamCodec::H264:
            return "rtph264depay";
        case StreamCodec::H265:
            return "rtph265depay";
        default:
            return "rtph264depay";
    }
}

std::string GStreamerRTSPHandler::getParserElement() const {
    switch (stream_codec_) {
        case StreamCodec::H264:
            return "h264parse";
        case StreamCodec::H265:
            return "h265parse";
        default:
            return "h264parse";
    }
}

bool GStreamerRTSPHandler::initializeGStreamer() {
    GError* error = nullptr;
    std::string pipeline_str = buildNvidiaHardwarePipeline();
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_ || error) {
        LOG_ERROR("GStreamer pipeline failed: {}", error ? error->message : "Unknown");
        if (error) g_error_free(error);
        return false;
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        LOG_ERROR("Appsink element not found");
        return false;
    }

    g_object_set(appsink_, "emit-signals", TRUE, "sync", FALSE, "max-buffers", 2, "drop", TRUE, nullptr);
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(onNewSample), this);

    bus_ = gst_element_get_bus(pipeline_);
    gst_bus_add_signal_watch(bus_);
    g_signal_connect(bus_, "message::error", G_CALLBACK(onBusMessage), this);
    g_signal_connect(bus_, "message::warning", G_CALLBACK(onBusMessage), this);
    g_signal_connect(bus_, "message::info", G_CALLBACK(onBusMessage), this);

    main_loop_ = g_main_loop_new(nullptr, FALSE);
    return true;
}

void GStreamerRTSPHandler::cleanupGStreamer() {
    if (bus_) {
        gst_bus_remove_signal_watch(bus_);
        gst_object_unref(bus_);
        bus_ = nullptr;
    }

    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

    if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
    }

    if (main_loop_) {
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr;
    }
}

void GStreamerRTSPHandler::gstreamerLoop() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    }

    if (main_loop_) {
        g_main_loop_run(main_loop_);
    }
}

void GStreamerRTSPHandler::captureLoop() {
    while (is_active_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void GStreamerRTSPHandler::processFrame(const cv::Mat& frame, uint64_t timestamp_ms) {
    if (!is_active_) return;

    // Track first frame timestamp
    if (current_clip_.empty()) {
        clip_start_timestamp_ms_ = timestamp_ms;
    }

    current_clip_.push_back(frame);
    clip_end_timestamp_ms_ = timestamp_ms;

    if (current_clip_.size() >= frames_per_clip_) {
        ClipContainer clip("clip_" + std::to_string(clip_start_timestamp_ms_),
                         camera_id_, current_clip_,
                         clip_start_timestamp_ms_, clip_end_timestamp_ms_);

        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (clip_queue_.size() < max_queue_size_) {
            clip_queue_.push(std::move(clip));
            queue_cv_.notify_one();
        }

        current_clip_.clear();
        clip_start_time_ = std::chrono::steady_clock::now();
    }
}

GstFlowReturn GStreamerRTSPHandler::onNewSample(GstElement* appsink, gpointer user_data) {
    GStreamerRTSPHandler* handler = static_cast<GStreamerRTSPHandler*>(user_data);

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) {
        return GST_FLOW_ERROR;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);

    if (buffer && caps) {
        // Extract relative timestamp from GStreamer buffer (PTS = Presentation Timestamp)
        GstClockTime pts = GST_BUFFER_PTS(buffer);
        uint64_t relative_timestamp_ms = GST_TIME_AS_MSECONDS(pts);

        // Initialize time offset on first frame
        if (handler->stream_start_pts_ms_ == 0) {
            handler->stream_start_pts_ms_ = relative_timestamp_ms;
            handler->stream_start_system_time_ = std::chrono::system_clock::now();
        }

        // Convert relative timestamp to absolute UTC timestamp
        auto elapsed = std::chrono::milliseconds(relative_timestamp_ms - handler->stream_start_pts_ms_);
        auto absolute_time = handler->stream_start_system_time_ + elapsed;
        uint64_t absolute_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            absolute_time.time_since_epoch()
        ).count();

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            int width, height;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);

            cv::Mat frame(height, width, CV_8UC3, map.data);
            handler->processFrame(frame.clone(), absolute_timestamp_ms);
            gst_buffer_unmap(buffer, &map);
        }
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

gboolean GStreamerRTSPHandler::onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data) {
    GStreamerRTSPHandler* handler = static_cast<GStreamerRTSPHandler*>(user_data);

    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR:
            handler->handlePipelineError(message);
            break;
        case GST_MESSAGE_WARNING:
            handler->handlePipelineWarning(message);
            break;
        case GST_MESSAGE_INFO:
            handler->handlePipelineInfo(message);
            break;
        default:
            break;
    }

    return TRUE;
}

void GStreamerRTSPHandler::handlePipelineError(GstMessage* message) {
    GError* error = nullptr;
    gchar* debug_info = nullptr;

    gst_message_parse_error(message, &error, &debug_info);
    LOG_ERROR("GStreamer error: {}", error ? error->message : "Unknown");

    if (error) g_error_free(error);
    if (debug_info) g_free(debug_info);

    is_active_ = false;
}

void GStreamerRTSPHandler::handlePipelineWarning(GstMessage* message) {
    GError* warning = nullptr;
    gchar* debug_info = nullptr;

    gst_message_parse_warning(message, &warning, &debug_info);

    if (warning) g_error_free(warning);
    if (debug_info) g_free(debug_info);
}

void GStreamerRTSPHandler::handlePipelineInfo(GstMessage* message) {
    GError* info = nullptr;
    gchar* debug_info = nullptr;

    gst_message_parse_info(message, &info, &debug_info);

    if (info) g_error_free(info);
    if (debug_info) g_free(debug_info);
}

OpenCVFileHandler::OpenCVFileHandler(int clip_length)
    : is_active_(false), clip_length_(clip_length), current_frame_index_(0), fps_(0), total_frames_(0) {}

OpenCVFileHandler::~OpenCVFileHandler() {
    stopStream();
}

bool OpenCVFileHandler::startStream(const std::string& file_path) {
    if (is_active_) {
        return false;
    }

    file_path_ = file_path;

    if (camera_id_.empty()) {
        size_t last_slash = file_path.find_last_of("/\\");
        camera_id_ = (last_slash != std::string::npos) ? file_path.substr(last_slash + 1) : file_path;
    }

    capture_.open(file_path_);
    if (!capture_.isOpened()) {
        LOG_ERROR("Cannot open video file: {}", file_path_);
        return false;
    }

    fps_ = capture_.get(cv::CAP_PROP_FPS);
    frames_per_clip_ = clip_length_ * fps_;
    total_frames_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
    current_frame_index_ = 0;
    is_active_ = true;

    return true;
}

void OpenCVFileHandler::stopStream() {
    if (!is_active_) return;

    is_active_ = false;
    if (capture_.isOpened()) {
        capture_.release();
    }
}

std::optional<ClipContainer> OpenCVFileHandler::getNextClip() {
    if (!is_active_ || !capture_.isOpened()) {
        return std::nullopt;
    }

    std::vector<cv::Mat> clip_frames;
    int frames_read = 0;

    while (frames_read < frames_per_clip_ && current_frame_index_ < total_frames_) {
        cv::Mat frame;
        if (capture_.read(frame)) {
            clip_frames.push_back(frame.clone());
            current_frame_index_++;
            frames_read++;
        } else {
            break;
        }
    }

    if (clip_frames.empty()) {
        is_active_ = false;
        return std::nullopt;
    }

    uint64_t start_timestamp_ms = (fps_ > 0) ?
        static_cast<uint64_t>(((current_frame_index_ - frames_read) / fps_) * 1000.0) : 0;
    uint64_t end_timestamp_ms = (fps_ > 0) ?
        static_cast<uint64_t>((current_frame_index_ / fps_) * 1000.0) : 0;

    ClipContainer clip("clip_" + std::to_string(current_frame_index_),
                      camera_id_, clip_frames, start_timestamp_ms, end_timestamp_ms);

    return clip;
}

bool OpenCVFileHandler::isActive() const {
    return is_active_ && current_frame_index_ < total_frames_;
}

}