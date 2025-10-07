#include "../include/object_detectors.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <random>

namespace vision_analysis {

BaseObjectDetector::BaseObjectDetector(const std::string& model_path, float conf_thresh, float nms_thresh)
    : model_path_(model_path), confidence_threshold_(conf_thresh), nms_threshold_(nms_thresh), next_track_id_(1) {}

std::vector<TrackedObject> BaseObjectDetector::detectAndTrack(const SampledFrames& frames) {
    std::vector<std::vector<DetectionResult>> all_detections;

    for (const auto& frame : frames.sampled_frames) {
        std::vector<DetectionResult> frame_detections = detectObjects(frame);
        all_detections.push_back(frame_detections);
    }

    return performTracking(all_detections, frames);
}

std::vector<TrackedObject> BaseObjectDetector::performTracking(
    const std::vector<std::vector<DetectionResult>>& all_detections,
    const SampledFrames& frames) {

    std::vector<TrackedObject> tracked_objects;
    std::unordered_map<std::string, TrackedObject> active_tracks;

    for (size_t frame_idx = 0; frame_idx < all_detections.size(); ++frame_idx) {
        const auto& detections = all_detections[frame_idx];

        for (const auto& detection : detections) {
            std::string track_id = assignTrackId(detection);

            auto it = active_tracks.find(track_id);
            if (it == active_tracks.end()) {
                TrackedObject new_track(generateObjectId(), track_id,
                                      frames.clip_id, frames.camera_id,
                                      detection.class_name, frames.timestamp);
                new_track.bbox_history.push_back(detection.bbox);
                new_track.confidence_scores.push_back(detection.confidence);
                active_tracks[track_id] = new_track;
            } else {
                it->second.bbox_history.push_back(detection.bbox);
                it->second.confidence_scores.push_back(detection.confidence);
            }
        }
    }

    for (auto& [track_id, track] : active_tracks) {
        tracked_objects.push_back(track);
    }

    return tracked_objects;
}

std::string BaseObjectDetector::generateObjectId() {
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    std::stringstream ss;
    ss << "obj_" << timestamp << "_" << dis(gen);
    return ss.str();
}

std::string BaseObjectDetector::assignTrackId(const DetectionResult& detection) {
    std::stringstream ss;
    ss << detection.class_name << "_" << next_track_id_++;
    return ss.str();
}

YOLODetector::YOLODetector(const std::string& model_path)
    : BaseObjectDetector(model_path), model_loaded_(false) {
    loadModel();
}

void YOLODetector::loadModel() {
    std::cout << "Loading YOLO model from: " << model_path_ << std::endl;
    model_loaded_ = true;
}

std::vector<DetectionResult> YOLODetector::detectObjects(const cv::Mat& frame) {
    std::vector<DetectionResult> detections;

    if (!model_loaded_ || frame.empty()) {
        return detections;
    }

    BoundingBox bbox(100, 100, 200, 150);
    detections.emplace_back(bbox, 0.85f, "person", 0);

    BoundingBox bbox2(300, 200, 150, 100);
    detections.emplace_back(bbox2, 0.72f, "car", 2);

    return detections;
}

SSDDetector::SSDDetector(const std::string& model_path)
    : BaseObjectDetector(model_path), model_loaded_(false) {
    loadModel();
}

void SSDDetector::loadModel() {
    std::cout << "Loading SSD model from: " << model_path_ << std::endl;
    model_loaded_ = true;
}

std::vector<DetectionResult> SSDDetector::detectObjects(const cv::Mat& frame) {
    std::vector<DetectionResult> detections;

    if (!model_loaded_ || frame.empty()) {
        return detections;
    }

    BoundingBox bbox(80, 90, 180, 160);
    detections.emplace_back(bbox, 0.78f, "person", 0);

    return detections;
}

FasterRCNNDetector::FasterRCNNDetector(const std::string& model_path)
    : BaseObjectDetector(model_path), model_loaded_(false) {
    loadModel();
}

void FasterRCNNDetector::loadModel() {
    std::cout << "Loading Faster R-CNN model from: " << model_path_ << std::endl;
    model_loaded_ = true;
}

std::vector<DetectionResult> FasterRCNNDetector::detectObjects(const cv::Mat& frame) {
    std::vector<DetectionResult> detections;

    if (!model_loaded_ || frame.empty()) {
        return detections;
    }

    BoundingBox bbox(120, 110, 190, 140);
    detections.emplace_back(bbox, 0.91f, "person", 0);

    BoundingBox bbox2(320, 180, 160, 120);
    detections.emplace_back(bbox2, 0.68f, "bicycle", 1);

    return detections;
}

}