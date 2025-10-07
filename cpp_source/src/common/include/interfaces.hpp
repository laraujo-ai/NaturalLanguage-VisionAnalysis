#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <chrono>

namespace nl_video_analysis {

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

}
// struct BoundingBox {
//     float x, y, width, height;
//     BoundingBox(float x = 0, float y = 0, float w = 0, float h = 0) : x(x), y(y), width(w), height(h) {}
// };

// struct TrackedObject {
//     std::string object_id;
//     std::string track_id;
//     std::string clip_id;
//     std::string camera_id;
//     std::vector<BoundingBox> bbox_history;
//     std::vector<float> confidence_scores;
//     std::string class_name;
//     std::vector<float> embedding_history;
//     std::vector<float> best_frame_embedding;
//     double timestamp;

//     TrackedObject(const std::string& obj_id, const std::string& tr_id, const std::string& cl_id,
//                   const std::string& cam_id, const std::string& cls_name, double ts)
//         : object_id(obj_id), track_id(tr_id), clip_id(cl_id), camera_id(cam_id),
//           class_name(cls_name), timestamp(ts) {}
// };


// class IObjectDetector {
// public:
//     virtual ~IObjectDetector() = default;
//     virtual std::vector<TrackedObject> detectAndTrack(const ClipContainer& clip) = 0;
// };

// class IStorageHandler {
// public:
//     virtual ~IStorageHandler() = default;
//     virtual std::string saveClip(const ClipContainer& clip, const std::string& path) = 0;
//     virtual bool saveEmbeddings(const std::vector<TrackedObject>& objects) = 0;
// };

// }