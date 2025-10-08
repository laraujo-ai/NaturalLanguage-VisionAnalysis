#ifndef SORT_TRACKER_H
#define SORT_TRACKER_H

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <ctime>

#include "../../../common/include/interfaces.hpp"

namespace nl_vision_analysis
{
constexpr int MAX_HISTORY_SIZE = 200;

class GeneralTracklet;

class KalmanFilter {
public:
    KalmanFilter(int dim_x, int dim_z);

    void predict();
    void update(const Eigen::VectorXd& z);

    Eigen::MatrixXd F;  // State transition matrix
    Eigen::MatrixXd H;  // Measurement matrix
    Eigen::MatrixXd Q;  // Process noise covariance
    Eigen::MatrixXd R;  // Measurement noise covariance
    Eigen::MatrixXd P;  // State covariance matrix
    Eigen::VectorXd x;  // State vector

private:
    int dim_x_;
    int dim_z_;
    Eigen::MatrixXd I;
};

class GeneralTracklet : public nl_video_analysis::BaseTracklet {
public:
    GeneralTracklet(const Eigen::Vector4d& bbox, double conf, int label);

    void update(const Eigen::Vector4d& bbox, double conf) override;
    Eigen::Vector4d predict() override;
    Eigen::Vector4d get_state() const override;
    nlohmann::json to_json() const;

    int time_since_update;
    std::string id;
    std::vector<Eigen::Vector4d> history;
    int hits;
    int hit_streak;
    int age;
    int64_t tracker_id;
    double conf;
    int label;

private:
    std::unique_ptr<KalmanFilter> kf;
};

class SortTracker {
public:
    SortTracker(int max_age = 1, int min_hits = 3, double iou_threshold = 0.3);

    std::vector<nlohmann::json> track(const std::vector<nl_video_analysis::Detection>& dets);

private:
    int max_age_;
    int min_hits_;
    double iou_threshold_;
    std::vector<std::unique_ptr<GeneralTracklet>> trackers_;
    int frame_count_;
};

std::string generate_ulid();
Eigen::Vector4d convert_bbox_to_z(const Eigen::Vector4d& bbox);
Eigen::Vector4d convert_x_to_bbox(const Eigen::VectorXd& x);
Eigen::MatrixXd iou_batch(const Eigen::MatrixXd& bb_test, const Eigen::MatrixXd& bb_gt);
std::tuple<std::vector<std::pair<int,int>>, std::vector<int>, std::vector<int>>
    associate_detections_to_trackers(const std::vector<nl_video_analysis::Detection>& detections,
                                      const std::vector<Eigen::Vector4d>& trackers,
                                      double iou_threshold);
std::vector<std::pair<int,int>> linear_assignment(const Eigen::MatrixXd& cost_matrix);

#endif // SORT_TRACKER_H

}

