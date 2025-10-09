#include "components/video_analysis_engine/include/VideoAnalysisEngine.hpp"
#include "common/include/config_parser.hpp"
#include "common/include/logger.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>

std::atomic<bool> running(true);

void signalHandler(int signum) {
    LOG_INFO("Interrupt signal received, shutting down...");
    running = false;
}

int main(int argc, char* argv[])
{
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " config.json" << std::endl;
        return 1;
    }

    std::string config_file = argv[1];

    LOG_INFO("=== Natural Language Vision Analysis System ===");
    LOG_INFO("Loading configuration from: {}", config_file);

    nl_video_analysis::VideoAnalysisConfig config;
    try {
        config = nl_video_analysis::ConfigParser::parseFromFile(config_file);
        LOG_INFO("Configuration loaded: {} cameras, {} sampled frames per clip",
                 config.cameras.size(), config.sampled_frames_count);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load config: {}", e.what());
        return 1;
    }

    if (config.cameras.empty()) {
        LOG_ERROR("No cameras configured");
        return 1;
    }

    nl_video_analysis::VideoAnalysisEngine processor(config);

    for (const auto& camera : config.cameras) {
        if (!processor.addSource(camera.source_url, camera.camera_id, camera.source_type, camera.stream_codec)) {
            LOG_WARN("Failed to add camera: {}", camera.camera_id);
        }
    }

    processor.start();

    if (!processor.isRunning()) {
        LOG_ERROR("Failed to start pipeline");
        return 1;
    }

    LOG_INFO("Processing started. Press Ctrl+C to stop");

    auto start_time = std::chrono::steady_clock::now();

    while (running && processor.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    processor.stop();

    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time).count();

    LOG_INFO("=== Processing Summary ===");
    LOG_INFO("Total runtime: {} seconds", total_time);

    return 0;
}
