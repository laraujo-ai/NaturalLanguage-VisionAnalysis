#pragma once

#include "../../../common/include/interfaces.hpp"
#include "../../../common/include/base_model.cpp"
#include <opencv2/opencv.hpp>

namespace vision_analysis {

struct YOLODetection {
    float x, y, width, height;
    float confidence;
    int class_id;
    std::string class_name;
};

struct CLIPEmbedding {
    std::vector<float> features;
    int feature_size;

    CLIPEmbedding() : feature_size(0) {}
    CLIPEmbedding(const std::vector<float>& feats) : features(feats), feature_size(feats.size()) {}
};

class YOLOObjectDetector : public IBaseModel<cv::Mat, std::vector<YOLODetection>> {
private:
    float confidence_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
    int input_width_;
    int input_height_;

    std::vector<std::string> loadClassNames(const std::string& class_file);
    std::vector<YOLODetection> performNMS(const std::vector<YOLODetection>& detections);

protected:
    std::vector<Ort::Value> preprocess(const cv::Mat& input) override;
    std::vector<YOLODetection> postprocess(std::vector<Ort::Value>& output_tensors) override;

public:
    YOLOObjectDetector(const std::string& model_path,
                       const std::string& class_file = "",
                       int num_threads = 4,
                       float conf_thresh = 0.5f,
                       float nms_thresh = 0.4f);

    void setInputSize(int width, int height) { input_width_ = width; input_height_ = height; }
    void setThresholds(float conf_thresh, float nms_thresh) {
        confidence_threshold_ = conf_thresh;
        nms_threshold_ = nms_thresh;
    }
};

class CLIPEmbeddingModel : public IBaseModel<cv::Mat, CLIPEmbedding> {
private:
    int input_width_;
    int input_height_;
    int embedding_size_;

protected:
    std::vector<Ort::Value> preprocess(const cv::Mat& input) override;
    CLIPEmbedding postprocess(std::vector<Ort::Value>& output_tensors) override;

public:
    CLIPEmbeddingModel(const std::string& model_path,
                       int num_threads = 4,
                       int input_size = 224,
                       int embedding_size = 512);

    CLIPEmbedding generateTrackEmbedding(const std::vector<cv::Mat>& track_frames);
    CLIPEmbedding generateBestFrameEmbedding(const cv::Mat& best_frame);
};

class VisionAnalysisDetector : public IObjectDetector {
private:
    std::unique_ptr<YOLOObjectDetector> object_detector_;
    std::unique_ptr<CLIPEmbeddingModel> clip_model_;
    std::unordered_map<std::string, std::vector<cv::Mat>> track_history_;
    int next_track_id_;

    std::string generateObjectId();
    std::string assignTrackId(const YOLODetection& detection, const cv::Mat& frame);
    cv::Mat getBestFrame(const std::vector<cv::Mat>& frames, const std::vector<float>& confidences);

public:
    VisionAnalysisDetector(const std::string& yolo_model_path,
                           const std::string& clip_model_path,
                           const std::string& class_file = "",
                           int num_threads = 4);

    std::vector<TrackedObject> detectAndTrack(const SampledFrames& frames) override;

    void setYOLOThresholds(float conf_thresh, float nms_thresh);
    void setYOLOInputSize(int width, int height);
};

}