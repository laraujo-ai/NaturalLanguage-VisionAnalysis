#pragma once

#include "../../../common/include/interfaces.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>

namespace nl_video_analysis {

enum class StreamCodec {
    H264,
    H265
};

class GStreamerRTSPHandler : public IStreamHandler {
private:
    std::string rtsp_url_;
    std::string camera_id_;
    std::atomic<bool> is_active_;
    std::thread capture_thread_;
    std::queue<ClipContainer> clip_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    int max_queue_size_;
    int frames_per_clip_; 
    int clip_length_; // this will be in seconds
    int target_fps_;
    int target_width_;
    int target_height_;
    StreamCodec stream_codec_;

    GstElement* pipeline_;
    GstElement* appsink_;
    GstBus* bus_;
    GMainLoop* main_loop_;
    std::thread gst_thread_;

    void gstreamerLoop();
    void captureLoop();
    bool initializeGStreamer();
    void cleanupGStreamer();
    std::string buildNvidiaHardwarePipeline() const;
    std::string getDepayElement() const;
    std::string getParserElement() const;

    std::vector<cv::Mat> current_clip_;
    std::chrono::steady_clock::time_point clip_start_time_;
    uint64_t clip_start_timestamp_ms_;
    uint64_t clip_end_timestamp_ms_;

    // For converting relative stream timestamps to absolute UTC
    std::chrono::system_clock::time_point stream_start_system_time_;
    uint64_t stream_start_pts_ms_;

    void processFrame(const cv::Mat& frame, uint64_t timestamp_ms);

    static GstFlowReturn onNewSample(GstElement* appsink, gpointer user_data);
    static gboolean onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data);
    void handlePipelineError(GstMessage* message);
    void handlePipelineWarning(GstMessage* message);
    void handlePipelineInfo(GstMessage* message);

public:
    GStreamerRTSPHandler(int clip_length = 5, int max_queue_size = 10,
                         int target_fps = 30, int target_width = 640, int target_height = 640,
                         StreamCodec codec = StreamCodec::H264);
    ~GStreamerRTSPHandler();

    bool startStream(const std::string& rtsp_url) override;
    void stopStream() override;
    std::optional<ClipContainer> getNextClip() override;
    bool isActive() const override;

    void setCameraId(const std::string& camera_id) { camera_id_ = camera_id; }
    void setFramesPerClip(int frames) { frames_per_clip_ = frames; }
    void setMaxQueueSize(int size) { max_queue_size_ = size; }
    void setTargetResolution(int width, int height) { target_width_ = width; target_height_ = height; }
    void setTargetFPS(int fps) { target_fps_ = fps; }
    void setStreamCodec(StreamCodec codec) { stream_codec_ = codec; }
};

class OpenCVFileHandler : public IStreamHandler {
private:
    std::string file_path_;
    std::string camera_id_;
    std::atomic<bool> is_active_;
    cv::VideoCapture capture_;

    int frames_per_clip_;
    int current_frame_index_;
    double fps_;
    int total_frames_;

public:
    OpenCVFileHandler(int frames_per_clip = 30);
    ~OpenCVFileHandler();

    bool startStream(const std::string& file_path) override;
    void stopStream() override;
    std::optional<ClipContainer> getNextClip() override;
    bool isActive() const override;

    void setCameraId(const std::string& camera_id) { camera_id_ = camera_id; }
    void setFramesPerClip(int frames) { frames_per_clip_ = frames; }

    double getFPS() const { return fps_; }
    int getTotalFrames() const { return total_frames_; }
    int getCurrentFrame() const { return current_frame_index_; }
};

enum class StreamSourceType {
    RTSP_STREAM,
    VIDEO_FILE
};

class StreamHandlerFactory {
public:
    static std::unique_ptr<IStreamHandler> createHandler(StreamSourceType type);
    static StreamSourceType detectSourceType(const std::string& source);
    static std::unique_ptr<IStreamHandler> createAutoDetect(const std::string& source);
};

}