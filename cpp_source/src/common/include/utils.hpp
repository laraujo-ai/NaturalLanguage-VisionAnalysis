#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <optional>
#include <algorithm>

namespace nl_video_analysis {
inline std::optional<cv::Mat> crop_object(const cv::Mat& frame,
                                          float x1, float y1,
                                          float x2, float y2,
                                          int padding = 10) {
    int h = frame.rows;
    int w = frame.cols;

    int x1_int = static_cast<int>(std::max(0.0f, x1));
    int y1_int = static_cast<int>(std::max(0.0f, y1));
    int x2_int = static_cast<int>(std::min(static_cast<float>(w), x2));
    int y2_int = static_cast<int>(std::min(static_cast<float>(h), y2));

    if (x2_int <= x1_int || y2_int <= y1_int) {
        return std::nullopt;
    }

    int y1_padded = std::max(0, y1_int - padding);
    int y2_padded = std::min(h, y2_int + padding);
    int x1_padded = std::max(0, x1_int - padding);
    int x2_padded = std::min(w, x2_int + padding);

    if (x2_padded <= x1_padded || y2_padded <= y1_padded) {
        return std::nullopt;
    }

    cv::Rect roi(x1_padded, y1_padded,
                 x2_padded - x1_padded,
                 y2_padded - y1_padded);

    cv::Mat cropped = frame(roi).clone();

    if (cropped.empty()) {
        return std::nullopt;
    }

    return cropped;
}

} 

#endif // UTILS_HPP

