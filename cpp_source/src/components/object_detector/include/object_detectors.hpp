#pragma once

#include "../../../common/include/interfaces.hpp"
#include <unordered_map>
#include <memory>

namespace vision_analysis {

struct DetectionResult {
    BoundingBox bbox;
    float confidence;
    std::string class_name;
    int class_id;

    DetectionResult(const BoundingBox& bb, float conf, const std::string& cls, int id)
        : bbox(bb), confidence(conf), class_name(cls), class_id(id) {}
};

class BaseObjectDetector : public IObjectDetector {
protected:
    std::string model_path_;
    float confidence_threshold_;
    float nms_threshold_;
    std::unordered_map<int, std::string> track_history_;
    int next_track_id_;

    virtual std::vector<DetectionResult> detectObjects(const cv::Mat& frame) = 0;
    std::vector<TrackedObject> performTracking(const std::vector<std::vector<DetectionResult>>& all_detections,
                                               const SampledFrames& frames);
    std::string generateObjectId();
    std::string assignTrackId(const DetectionResult& detection);

public:
    BaseObjectDetector(const std::string& model_path, float conf_thresh = 0.5, float nms_thresh = 0.4);
    virtual ~BaseObjectDetector() = default;

    std::vector<TrackedObject> detectAndTrack(const SampledFrames& frames) override;
};

class YOLODetector : public BaseObjectDetector {
private:
    bool model_loaded_;

    void loadModel();

public:
    YOLODetector(const std::string& model_path);
    ~YOLODetector() = default;

protected:
    std::vector<DetectionResult> detectObjects(const cv::Mat& frame) override;
};

class SSDDetector : public BaseObjectDetector {
private:
    bool model_loaded_;

    void loadModel();

public:
    SSDDetector(const std::string& model_path);
    ~SSDDetector() = default;

protected:
    std::vector<DetectionResult> detectObjects(const cv::Mat& frame) override;
};

class FasterRCNNDetector : public BaseObjectDetector {
private:
    bool model_loaded_;

    void loadModel();

public:
    FasterRCNNDetector(const std::string& model_path);
    ~FasterRCNNDetector() = default;

protected:
    std::vector<DetectionResult> detectObjects(const cv::Mat& frame) override;
};

}