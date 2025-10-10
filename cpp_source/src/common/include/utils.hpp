#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <optional>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nl_video_analysis {
inline std::optional<cv::Mat> crop_object(const cv::Mat& frame,
                                          int x1, int y1,
                                          int x2, int y2,
                                          int padding = 10) {
    int h = frame.rows;
    int w = frame.cols;

    int x1_int = std::max(0, x1);
    int y1_int = std::max(0, y1);
    int x2_int = std::min(w, x2);
    int y2_int = std::min(h, y2);

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

inline std::vector<float> averageTrackEmbeddings(const std::vector<std::vector<float>>& track_embeddings) {
    if (track_embeddings.empty()) {
        return std::vector<float>();
    }

    size_t embedding_dim = track_embeddings[0].size();

    for (const auto& embedding : track_embeddings) {
        if (embedding.size() != embedding_dim) {
            throw std::runtime_error("All embeddings must have the same dimension for average pooling");
        }
    }

    std::vector<float> averaged_embedding(embedding_dim, 0.0f);

    for (const auto& embedding : track_embeddings) {
        for (size_t i = 0; i < embedding_dim; ++i) {
            averaged_embedding[i] += embedding[i];
        }
    }

    float num_embeddings = static_cast<float>(track_embeddings.size());
    for (float& val : averaged_embedding) {
        val /= num_embeddings;
    }

    float norm = 0.0f;
    for (float val : averaged_embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-6f) {
        for (float& val : averaged_embedding) {
            val /= norm;
        }
    }

    return averaged_embedding;
}

} 

#endif // UTILS_HPP

