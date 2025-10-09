#define CATCH_CONFIG_MAIN
#include "../../../lib/catch2/catch_amalgamated.hpp"
#include "vision_stream_handlers.hpp"
#include "../../../common/include/interfaces.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace nl_video_analysis;

std::string createTestVideo(const std::string& filename, int num_frames, double fps = 30.0) {
    std::string filepath = "/tmp/" + filename;

    cv::VideoWriter writer(filepath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          fps, cv::Size(640, 480));

    if (!writer.isOpened()) {
        throw std::runtime_error("Failed to create test video");
    }

    for (int i = 0; i < num_frames; ++i) {
        cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame, std::to_string(i), cv::Point(50, 50),
                   cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
        writer.write(frame);
    }

    writer.release();
    return filepath;
}

TEST_CASE("OpenCVFileHandler basic functionality", "[stream_handler]") {
    std::string test_video = createTestVideo("test_basic.avi", 100);

    SECTION("Successfully starts stream with valid video file") {
        OpenCVFileHandler handler(5);
        REQUIRE(handler.startStream(test_video));
        REQUIRE(handler.isActive());
        handler.stopStream();
    }

    SECTION("Fails to start stream with invalid file path") {
        OpenCVFileHandler handler(5);
        REQUIRE_FALSE(handler.startStream("/nonexistent/video.mp4"));
        REQUIRE_FALSE(handler.isActive());
    }

    SECTION("Cannot start already active stream") {
        OpenCVFileHandler handler(5);
        REQUIRE(handler.startStream(test_video));
        REQUIRE_FALSE(handler.startStream(test_video)); // Second start should fail
        handler.stopStream();
    }

    SECTION("Stop stream when not active is safe") {
        OpenCVFileHandler handler(5);
        handler.stopStream(); 
        REQUIRE_FALSE(handler.isActive());
    }

    std::remove(test_video.c_str());
}

TEST_CASE("OpenCVFileHandler clip generation", "[stream_handler]") {
    std::string test_video = createTestVideo("test_clips.avi", 150, 30.0);

    SECTION("Generates clips with correct frame count") {
        OpenCVFileHandler handler(5);
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());

        // At 30 fps, 5 seconds = 150 frames
        REQUIRE(clip->frames.size() == 150);
        handler.stopStream();
    }

    SECTION("Generates multiple sequential clips") {
        OpenCVFileHandler handler(1); 
        handler.startStream(test_video);

        auto clip1 = handler.getNextClip();
        REQUIRE(clip1.has_value());
        REQUIRE(clip1->frames.size() == 30);

        auto clip2 = handler.getNextClip();
        REQUIRE(clip2.has_value());
        REQUIRE(clip2->frames.size() == 30);

        REQUIRE(clip1->clip_id != clip2->clip_id);
        handler.stopStream();
    }

    SECTION("Returns nullopt when video ends") {
        OpenCVFileHandler handler(10);
        handler.startStream(test_video);

        auto clip1 = handler.getNextClip();
        REQUIRE(clip1.has_value());

        auto clip2 = handler.getNextClip();
        REQUIRE_FALSE(clip2.has_value());
        REQUIRE_FALSE(handler.isActive());
        handler.stopStream();
    }

    std::remove(test_video.c_str());
}

TEST_CASE("OpenCVFileHandler metadata handling", "[stream_handler]") {
    std::string test_video = createTestVideo("test_metadata.avi", 60, 30.0);

    SECTION("Auto-generates camera_id from filename") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());
        REQUIRE(clip->camera_id == "test_metadata.avi");
        handler.stopStream();
    }

    SECTION("Uses custom camera_id when set") {
        OpenCVFileHandler handler(1);
        handler.setCameraId("custom_camera_001");
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());
        REQUIRE(clip->camera_id == "custom_camera_001");
        handler.stopStream();
    }

    SECTION("Clip contains valid timestamps") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());
        REQUIRE(clip->start_timestamp_ms >= 0);
        REQUIRE(clip->end_timestamp_ms > clip->start_timestamp_ms);
        handler.stopStream();
    }

    SECTION("Each clip has unique clip_id") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        auto clip1 = handler.getNextClip();
        auto clip2 = handler.getNextClip();

        REQUIRE(clip1.has_value());
        REQUIRE(clip2.has_value());
        REQUIRE(clip1->clip_id != clip2->clip_id);
        handler.stopStream();
    }

    std::remove(test_video.c_str());
}

TEST_CASE("OpenCVFileHandler frame properties", "[stream_handler]") {
    std::string test_video = createTestVideo("test_frames.avi", 90, 30.0);

    SECTION("Frames are non-empty") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());

        for (const auto& frame : clip->frames) {
            REQUIRE_FALSE(frame.empty());
        }
        handler.stopStream();
    }

    SECTION("Frames have consistent dimensions") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());

        cv::Size expected_size = clip->frames[0].size();
        for (const auto& frame : clip->frames) {
            REQUIRE(frame.size() == expected_size);
        }
        handler.stopStream();
    }

    SECTION("Frames have correct type") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());

        for (const auto& frame : clip->frames) {
            REQUIRE(frame.type() == CV_8UC3);
        }
        handler.stopStream();
    }

    std::remove(test_video.c_str());
}

TEST_CASE("OpenCVFileHandler configuration", "[stream_handler]") {
    std::string test_video = createTestVideo("test_config.avi", 120, 30.0);

    SECTION("Respects custom clip length") {
        OpenCVFileHandler handler(2); 
        handler.startStream(test_video);

        auto clip = handler.getNextClip();
        REQUIRE(clip.has_value());
        REQUIRE(clip->frames.size() == 60);
        handler.stopStream();
    }

    SECTION("Reads video properties correctly") {
        OpenCVFileHandler handler(1);
        handler.startStream(test_video);

        REQUIRE(handler.getFPS() == Catch::Approx(30.0).margin(0.1));
        REQUIRE(handler.getTotalFrames() == 120);
        REQUIRE(handler.getCurrentFrame() == 0);

        handler.getNextClip();
        REQUIRE(handler.getCurrentFrame() == 30);
        handler.stopStream();
    }

    std::remove(test_video.c_str());
}
