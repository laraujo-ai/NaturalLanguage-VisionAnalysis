#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cmath>

namespace nl_video_analysis {

struct StageMetrics {
    size_t count = 0;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    std::vector<double> samples;  // For percentile calculation

    void addSample(double duration_ms) {
        count++;
        total_ms += duration_ms;
        min_ms = std::min(min_ms, duration_ms);
        max_ms = std::max(max_ms, duration_ms);
        samples.push_back(duration_ms);
    }

    double getAverage() const {
        return count > 0 ? total_ms / count : 0.0;
    }

    double getPercentile(double percentile) const {
        if (samples.empty()) return 0.0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(percentile * sorted.size());
        return sorted[std::min(idx, sorted.size() - 1)];
    }

    void reset() {
        count = 0;
        total_ms = 0.0;
        min_ms = std::numeric_limits<double>::max();
        max_ms = 0.0;
        samples.clear();
    }
};

class PipelineBenchmark {
public:
    static PipelineBenchmark& getInstance() {
        static PipelineBenchmark instance;
        return instance;
    }

    // Record timing for a stage
    void recordTiming(const std::string& stage_name, double duration_ms, const std::string& camera_id = "") {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::string key = camera_id.empty() ? stage_name : camera_id + ":" + stage_name;
        metrics_[key].addSample(duration_ms);

        // Also track global stage metrics
        if (!camera_id.empty()) {
            metrics_[stage_name].addSample(duration_ms);
        }
    }

    // Get metrics for a specific stage
    StageMetrics getMetrics(const std::string& stage_name, const std::string& camera_id = "") const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        std::string key = camera_id.empty() ? stage_name : camera_id + ":" + stage_name;

        auto it = metrics_.find(key);
        if (it != metrics_.end()) {
            return it->second;
        }
        return StageMetrics{};
    }

    // Get all metrics
    std::unordered_map<std::string, StageMetrics> getAllMetrics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        return metrics_;
    }

    // Reset all metrics
    void reset() {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.clear();
    }

    // Generate summary report
    std::string generateReport() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        std::string report = "\n=== Pipeline Benchmark Report ===\n";

        for (const auto& [stage_name, metrics] : metrics_) {
            if (metrics.count == 0) continue;

            report += "\n" + stage_name + ":\n";
            report += "  Count: " + std::to_string(metrics.count) + "\n";
            report += "  Average: " + std::to_string(metrics.getAverage()) + " ms\n";
            report += "  Min: " + std::to_string(metrics.min_ms) + " ms\n";
            report += "  Max: " + std::to_string(metrics.max_ms) + " ms\n";
            report += "  P50: " + std::to_string(metrics.getPercentile(0.50)) + " ms\n";
            report += "  P95: " + std::to_string(metrics.getPercentile(0.95)) + " ms\n";
            report += "  P99: " + std::to_string(metrics.getPercentile(0.99)) + " ms\n";
        }

        return report;
    }

private:
    PipelineBenchmark() = default;
    ~PipelineBenchmark() = default;
    PipelineBenchmark(const PipelineBenchmark&) = delete;
    PipelineBenchmark& operator=(const PipelineBenchmark&) = delete;

    mutable std::mutex metrics_mutex_;
    std::unordered_map<std::string, StageMetrics> metrics_;
};

// RAII-style timer for automatic timing
class ScopedTimer {
public:
    ScopedTimer(const std::string& stage_name, const std::string& camera_id = "")
        : stage_name_(stage_name), camera_id_(camera_id),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start_).count();
        PipelineBenchmark::getInstance().recordTiming(stage_name_, duration_ms, camera_id_);
    }

    // Get elapsed time without stopping timer
    double getElapsedMs() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::string stage_name_;
    std::string camera_id_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace nl_video_analysis
