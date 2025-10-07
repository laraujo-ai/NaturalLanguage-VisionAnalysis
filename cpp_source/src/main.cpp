#include "components/media_processing/include/MediaProcessor.hpp"
#include "common/include/config_parser.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>

std::atomic<bool> running(true);

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Shutting down..." << std::endl;
    running = false;
}

int main(int argc, char* argv[])
{
    // Register signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Check if config file is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " config.json" << std::endl;
        return 1;
    }

    std::string config_file = argv[1];
    std::cout << "============================================" << std::endl;
    std::cout << "  Natural Language Vision Analysis System  " << std::endl;
    std::cout << "  Stream Processing + Frame Sampling Demo  " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "\nLoading configuration from: " << config_file << std::endl;

    // Parse configuration
    nl_video_analysis::MediaProcessorConfig config;
    try {
        config = nl_video_analysis::ConfigParser::parseFromFile(config_file);
        std::cout << "\nConfiguration loaded successfully!" << std::endl;
        std::cout << "  - Max connections: " << config.max_connections << std::endl;
        std::cout << "  - Frames per clip: " << config.frames_per_clip << std::endl;
        std::cout << "  - Sampled frames: " << config.sampled_frames_count << std::endl;
        std::cout << "  - Sampler type: " << config.sampler_type << std::endl;
        std::cout << "  - Cameras configured: " << config.cameras.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return 1;
    }

    if (config.cameras.empty()) {
        std::cerr << "No cameras configured in config file. Exiting." << std::endl;
        return 1;
    }

    // Create MediaProcessor
    std::cout << "\nInitializing MediaProcessor..." << std::endl;
    nl_video_analysis::MediaProcessor processor(config);

    // Add camera sources from config
    std::cout << "\nAdding camera sources..." << std::endl;
    for (const auto& camera : config.cameras) {
        std::cout << "  - " << camera.camera_id << " (" << camera.source_type << "): "
                  << camera.source_url << std::endl;

        if (!processor.addSource(camera.source_url, camera.camera_id, camera.source_type)) {
            std::cerr << "    WARNING: Failed to add source: " << camera.camera_id << std::endl;
        }
    }

    // Start processing
    std::cout << "\nStarting media processor..." << std::endl;
    processor.start();

    if (!processor.isRunning()) {
        std::cerr << "Failed to start media processor. Exiting." << std::endl;
        return 1;
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Processing started! Press Ctrl+C to stop  " << std::endl;
    std::cout << "============================================\n" << std::endl;

    // Main loop - process clips with sampled frames
    int clip_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (running && processor.isRunning()) {
        nl_video_analysis::ClipContainer clip;

        if (processor.getNextClip(clip)) {
            clip_count++;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();

            std::cout << "\n[" << elapsed << "s] Received clip #" << clip_count << std::endl;
            std::cout << "  Camera ID: " << clip.camera_id << std::endl;
            std::cout << "  Clip ID: " << clip.clip_id << std::endl;
            std::cout << "  Total frames: " << clip.frames.size() << std::endl;
            std::cout << "  Sampled frames: " << clip.sampled_frames.size() << std::endl;
            std::cout << "  Timestamp: " << clip.start_timestamp_ms << " - "
                      << clip.end_timestamp_ms << " ms" << std::endl;

            if (!clip.sampled_frames.empty()) {
                const auto& first_frame = clip.sampled_frames[0];
                std::cout << "  Frame size: " << first_frame.cols << "x" << first_frame.rows << std::endl;
            }

            std::cout << "  Queue size: " << processor.getClipQueueSize() << std::endl;

            // Optional: You can save or display frames here
            // For example: cv::imwrite("frame_" + std::to_string(clip_count) + ".jpg", clip.sampled_frames[0]);

        } else {
            // No clips available, sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // Shutdown
    std::cout << "\n\nShutting down MediaProcessor..." << std::endl;
    processor.stop();

    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time).count();

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Processing Summary" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  Total runtime: " << total_time << " seconds" << std::endl;
    std::cout << "  Clips processed: " << clip_count << std::endl;
    if (total_time > 0) {
        std::cout << "  Average rate: " << (clip_count / total_time) << " clips/sec" << std::endl;
    }
    std::cout << "============================================\n" << std::endl;

    return 0;
}
