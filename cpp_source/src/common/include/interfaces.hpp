#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <Eigen/Dense>

namespace nl_video_analysis {


enum class StreamCodec {
    H264,
    H265
};


struct ClipContainer {
    std::string clip_id;
    std::string camera_id;
    std::string clip_path;

    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> sampled_frames;
    uint64_t start_timestamp_ms;
    uint64_t end_timestamp_ms;

    std::unordered_map<std::string, std::string> metadata;

    ClipContainer(const std::string& clip_id, const std::string& camera_id,
                  const std::vector<cv::Mat>& frames, uint64_t start_ts_ms, uint64_t end_ts_ms)
        : clip_id(clip_id), camera_id(camera_id), frames(frames),
          start_timestamp_ms(start_ts_ms), end_timestamp_ms(end_ts_ms) {}

    ClipContainer() = default;
};


class IStreamHandler {
    public:
    virtual ~IStreamHandler() = default;
    virtual bool startStream(const std::string& source_url) = 0;
    virtual void stopStream() = 0;
    virtual std::optional<ClipContainer> getNextClip() = 0;
    virtual bool isActive() const = 0;
};

class IFrameSampler {
    public:
    virtual ~IFrameSampler() = default;
    virtual void sampleFrames(ClipContainer& clip, int num_frames) = 0;
};

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

class BaseTracklet {
public:
    virtual ~BaseTracklet() = default;
    virtual void update(const Eigen::Vector4d& bbox, double conf) = 0;
    virtual Eigen::Vector4d predict() = 0;
    virtual Eigen::Vector4d get_state() const = 0;
};

}


// class IStorageHandler {
// public:
//     virtual ~IStorageHandler() = default;
//     virtual std::string saveClip(const ClipContainer& clip, const std::string& path) = 0;
//     virtual bool saveEmbeddings(const std::vector<TrackedObject>& objects) = 0;
// };

// }