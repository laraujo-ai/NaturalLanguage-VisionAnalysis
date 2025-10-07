#include "include/vision_stream_handlers.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {

    nl_video_analysis::GStreamerRTSPHandler handler(5, 5, 30, 640, 640, nl_video_analysis::StreamCodec::H264);
    handler.setCameraId("Office Entrance camera");

    std::string rtsp_url = "rtsp://vtviewer:Vtech123!@192.168.1.121/cam/realmonitor?channel=1&subtype=2";
    if (!handler.startStream(rtsp_url)) {
        std::cerr << "Failed to start RTSP stream!" << std::endl;
        return -1;
    }
    int clip_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(30)) {
        auto clip = handler.getNextClip();
        if (clip.has_value()) {
            clip_count++;
            std::cout << "Received clip #" << clip_count
                     << " - ID: " << clip->clip_id
                     << ", Camera: " << clip->camera_id
                     << ", Frames: " << clip->frames.size()
                     << ", Timestamp start: " << clip->start_timestamp_ms << "ms" << std::endl
                     << ", Timestamp end : " << clip->end_timestamp_ms << "ms" << std::endl;

            if (!clip->frames.empty()) {
                const auto& first_frame = clip->frames[0];
                std::cout << "  Frame size: " << first_frame.cols << "x" << first_frame.rows << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (!handler.isActive()) {
            std::cout << "Handler is no longer active" << std::endl;
            break;
        }
    }

    handler.stopStream();
    return 0;
}