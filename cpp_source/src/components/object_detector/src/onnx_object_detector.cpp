#include "../include/onnx_object_detector.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <opencv2/dnn.hpp>

namespace vision_analysis {

YOLOObjectDetector::YOLOObjectDetector(const std::string& model_path,
                                       const std::string& class_file,
                                       int num_threads,
                                       float conf_thresh,
                                       float nms_thresh)
    : IBaseModel<cv::Mat, std::vector<YOLODetection>>(model_path, num_threads),
      confidence_threshold_(conf_thresh),
      nms_threshold_(nms_thresh),
      input_width_(640),
      input_height_(640) {

    if (!class_file.empty()) {
        class_names_ = loadClassNames(class_file);
    } else {
        class_names_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"};
    }
}

std::vector<std::string> YOLOObjectDetector::loadClassNames(const std::string& class_file) {
    std::vector<std::string> names;
    std::ifstream file(class_file);
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            names.push_back(line);
        }
    }

    return names;
}

std::vector<Ort::Value> YOLOObjectDetector::preprocess(const cv::Mat& input) {
    cv::Mat resized, blob;
    cv::resize(input, resized, cv::Size(input_width_, input_height_));

    cv::dnn::blobFromImage(resized, blob, 1.0/255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false, CV_32F);

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    size_t input_tensor_size = 1 * 3 * input_height_ * input_width_;

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        (float*)blob.data,
        input_tensor_size,
        input_shape.data(),
        input_shape.size()
    ));

    return input_tensors;
}

std::vector<YOLODetection> YOLOObjectDetector::postprocess(std::vector<Ort::Value>& output_tensors) {
    std::vector<YOLODetection> detections;

    if (output_tensors.empty()) return detections;

    float* output = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int num_detections = output_shape[1];
    int detection_length = output_shape[2];

    for (int i = 0; i < num_detections; ++i) {
        float* detection = &output[i * detection_length];

        float confidence = detection[4];
        if (confidence < confidence_threshold_) continue;

        float max_class_score = 0.0f;
        int class_id = -1;

        for (int j = 5; j < detection_length; ++j) {
            if (detection[j] > max_class_score) {
                max_class_score = detection[j];
                class_id = j - 5;
            }
        }

        float final_confidence = confidence * max_class_score;
        if (final_confidence < confidence_threshold_) continue;

        YOLODetection det;
        det.x = detection[0] - detection[2] / 2.0f;
        det.y = detection[1] - detection[3] / 2.0f;
        det.width = detection[2];
        det.height = detection[3];
        det.confidence = final_confidence;
        det.class_id = class_id;
        det.class_name = (class_id < class_names_.size()) ? class_names_[class_id] : "unknown";

        detections.push_back(det);
    }

    return performNMS(detections);
}

std::vector<YOLODetection> YOLOObjectDetector::performNMS(const std::vector<YOLODetection>& detections) {
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto& det : detections) {
        boxes.push_back(cv::Rect(det.x, det.y, det.width, det.height));
        scores.push_back(det.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold_, nms_threshold_, indices);

    std::vector<YOLODetection> result;
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }

    return result;
}

CLIPEmbeddingModel::CLIPEmbeddingModel(const std::string& model_path,
                                       int num_threads,
                                       int input_size,
                                       int embedding_size)
    : IBaseModel<cv::Mat, CLIPEmbedding>(model_path, num_threads),
      input_width_(input_size),
      input_height_(input_size),
      embedding_size_(embedding_size) {}

std::vector<Ort::Value> CLIPEmbeddingModel::preprocess(const cv::Mat& input) {
    cv::Mat resized, normalized;
    cv::resize(input, resized, cv::Size(input_width_, input_height_));

    resized.convertTo(normalized, CV_32F, 1.0/255.0);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(normalized, mean, normalized);
    cv::divide(normalized, std, normalized);

    cv::Mat blob;
    cv::dnn::blobFromImage(normalized, blob, 1.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false, CV_32F);

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    size_t input_tensor_size = 1 * 3 * input_height_ * input_width_;

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        (float*)blob.data,
        input_tensor_size,
        input_shape.data(),
        input_shape.size()
    ));

    return input_tensors;
}

CLIPEmbedding CLIPEmbeddingModel::postprocess(std::vector<Ort::Value>& output_tensors) {
    if (output_tensors.empty()) return CLIPEmbedding();

    float* output = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int feature_size = output_shape[1];
    std::vector<float> features(output, output + feature_size);

    return CLIPEmbedding(features);
}

CLIPEmbedding CLIPEmbeddingModel::generateTrackEmbedding(const std::vector<cv::Mat>& track_frames) {
    if (track_frames.empty()) return CLIPEmbedding();

    std::vector<CLIPEmbedding> embeddings;
    for (const auto& frame : track_frames) {
        embeddings.push_back(run(frame));
    }

    std::vector<float> avg_embedding(embedding_size_, 0.0f);
    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < emb.features.size() && i < avg_embedding.size(); ++i) {
            avg_embedding[i] += emb.features[i];
        }
    }

    for (float& val : avg_embedding) {
        val /= embeddings.size();
    }

    return CLIPEmbedding(avg_embedding);
}

CLIPEmbedding CLIPEmbeddingModel::generateBestFrameEmbedding(const cv::Mat& best_frame) {
    return run(best_frame);
}

VisionAnalysisDetector::VisionAnalysisDetector(const std::string& yolo_model_path,
                                               const std::string& clip_model_path,
                                               const std::string& class_file,
                                               int num_threads)
    : next_track_id_(1) {

    object_detector_ = std::make_unique<YOLOObjectDetector>(yolo_model_path, class_file, num_threads);
    clip_model_ = std::make_unique<CLIPEmbeddingModel>(clip_model_path, num_threads);
}

std::vector<TrackedObject> VisionAnalysisDetector::detectAndTrack(const SampledFrames& frames) {
    std::vector<TrackedObject> tracked_objects;

    if (frames.sampled_frames.empty()) return tracked_objects;

    for (size_t frame_idx = 0; frame_idx < frames.sampled_frames.size(); ++frame_idx) {
        const cv::Mat& frame = frames.sampled_frames[frame_idx];
        std::vector<YOLODetection> detections = object_detector_->run(frame);

        for (const auto& detection : detections) {
            std::string track_id = assignTrackId(detection, frame);

            TrackedObject tracked_obj(generateObjectId(), track_id,
                                    frames.clip_id, frames.camera_id,
                                    detection.class_name, frames.timestamp);

            BoundingBox bbox(detection.x, detection.y, detection.width, detection.height);
            tracked_obj.bbox_history.push_back(bbox);
            tracked_obj.confidence_scores.push_back(detection.confidence);

            track_history_[track_id].push_back(frame);

            if (track_history_[track_id].size() >= 5) {
                CLIPEmbedding track_embedding = clip_model_->generateTrackEmbedding(track_history_[track_id]);
                tracked_obj.embedding_history = track_embedding.features;

                cv::Mat best_frame = getBestFrame(track_history_[track_id], tracked_obj.confidence_scores);
                CLIPEmbedding best_embedding = clip_model_->generateBestFrameEmbedding(best_frame);
                tracked_obj.best_frame_embedding = best_embedding.features;
            }

            tracked_objects.push_back(tracked_obj);
        }
    }

    return tracked_objects;
}

std::string VisionAnalysisDetector::generateObjectId() {
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    std::stringstream ss;
    ss << "obj_" << timestamp << "_" << dis(gen);
    return ss.str();
}

std::string VisionAnalysisDetector::assignTrackId(const YOLODetection& detection, const cv::Mat& frame) {
    std::stringstream ss;
    ss << detection.class_name << "_" << next_track_id_++;
    return ss.str();
}

cv::Mat VisionAnalysisDetector::getBestFrame(const std::vector<cv::Mat>& frames, const std::vector<float>& confidences) {
    if (frames.empty()) return cv::Mat();

    auto max_it = std::max_element(confidences.begin(), confidences.end());
    size_t max_idx = std::distance(confidences.begin(), max_it);

    return frames[std::min(max_idx, frames.size() - 1)];
}

void VisionAnalysisDetector::setYOLOThresholds(float conf_thresh, float nms_thresh) {
    object_detector_->setThresholds(conf_thresh, nms_thresh);
}

void VisionAnalysisDetector::setYOLOInputSize(int width, int height) {
    object_detector_->setInputSize(width, height);
}

}