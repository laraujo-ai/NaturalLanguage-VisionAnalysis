#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>

namespace nl_video_analysis {

struct CameraConfig {
    std::string camera_id;
    std::string source_url;
    std::string source_type;  // "rtsp" or "file" so far

    CameraConfig() = default;
    CameraConfig(const std::string& id, const std::string& url, const std::string& type)
        : camera_id(id), source_url(url), source_type(type) {}
};

struct ObjectDetectorConfig {
    std::string type;
    std::string weights_path;
    int number_of_threads;
    float conf_threshold;
    float nms_threshold;
    bool is_fp16;
};

struct TrackerConfig {

    int max_age;
    int min_hits;
    double iou_threshold;    
};

struct VideoAnalysisConfig {
    int max_connections = 10;
    int clip_length = 30;

    std::string sampler_type = "uniform";
    int sampled_frames_count = 5;

    int queue_max_size = 100;

    std::vector<CameraConfig> cameras;
    ObjectDetectorConfig object_detector;
    TrackerConfig tracker;

    
    int gst_buffer_size = 5;
    int gst_drop_frames = 5;
    int gst_target_fps = 30;
    int gst_frame_width = 640;
    int gst_frame_height = 640;

    VideoAnalysisConfig() = default;
};

class ConfigParser {
public:
    static VideoAnalysisConfig parseFromFile(const std::string& filepath);

private:
    static std::string trim(const std::string& str);
    static std::string removeQuotes(const std::string& str);
    static bool parseBool(const std::string& value);
    static int parseInt(const std::string& value);
    static std::string parseString(const std::string& value);
};

}
