#define CATCH_CONFIG_MAIN
#include "../../../lib/catch2/catch_amalgamated.hpp"
#include "frame_samplers.hpp"
#include "../../../common/include/interfaces.hpp"
#include <opencv2/opencv.hpp>

using namespace nl_video_analysis;

ClipContainer createTestClip(int frame_count, const std::string& camera_id = "test_camera") {
    ClipContainer clip;
    clip.camera_id = camera_id;
    clip.clip_id = "test_clip_001";

    for (int i = 0; i < frame_count; ++i) {
        cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame, std::to_string(i), cv::Point(50, 50),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        clip.frames.push_back(frame);
    }

    return clip;
}

TEST_CASE("UniformFrameSampler basic functionality", "[frame_sampler]") {
    UniformFrameSampler sampler;

    SECTION("Sample from clip with exact frame count") {
        ClipContainer clip = createTestClip(10);
        sampler.sampleFrames(clip, 5);

        REQUIRE(clip.sampled_frames.size() == 5);
        REQUIRE_FALSE(clip.sampled_frames[0].empty());
    }

    SECTION("Request more samples than available frames") {
        ClipContainer clip = createTestClip(3);
        sampler.sampleFrames(clip, 10);

        REQUIRE(clip.sampled_frames.size() == 3);
    }

    SECTION("Sample single frame") {
        ClipContainer clip = createTestClip(10);
        sampler.sampleFrames(clip, 1);

        REQUIRE(clip.sampled_frames.size() == 1);
    }

    SECTION("Empty clip handling") {
        ClipContainer clip = createTestClip(0);
        sampler.sampleFrames(clip, 5);

        REQUIRE(clip.sampled_frames.size() == 0);
    }

    SECTION("Zero samples requested") {
        ClipContainer clip = createTestClip(10);
        sampler.sampleFrames(clip, 0);

        REQUIRE(clip.sampled_frames.size() == 0);
    }
}

TEST_CASE("UniformFrameSampler uniform distribution", "[frame_sampler]") {
    UniformFrameSampler sampler;

    SECTION("Verify uniform spacing for 10 frames, 5 samples") {
        ClipContainer clip = createTestClip(10);
        sampler.sampleFrames(clip, 5);

        // With uniform sampling from 10 frames picking 5:
        // Expected indices: 0, 2, 4, 6, 8 (step = 10/5 = 2)
        REQUIRE(clip.sampled_frames.size() == 5);

        // Verify frames are actually different (check pixel values differ)
        for (size_t i = 1; i < clip.sampled_frames.size(); ++i) {
            double diff = cv::norm(clip.sampled_frames[i], clip.sampled_frames[i-1]);
            REQUIRE(diff > 0); 
        }
    }
}

TEST_CASE("UniformFrameSampler preserves frame properties", "[frame_sampler]") {
    UniformFrameSampler sampler;

    SECTION("Sampled frames retain original dimensions") {
        ClipContainer clip = createTestClip(10);
        cv::Size original_size = clip.frames[0].size();

        sampler.sampleFrames(clip, 5);

        for (const auto& frame : clip.sampled_frames) {
            REQUIRE(frame.size() == original_size);
        }
    }

    SECTION("Sampled frames retain original type") {
        ClipContainer clip = createTestClip(10);
        int original_type = clip.frames[0].type();

        sampler.sampleFrames(clip, 5);

        for (const auto& frame : clip.sampled_frames) {
            REQUIRE(frame.type() == original_type);
        }
    }
}

TEST_CASE("ClipContainer metadata", "[frame_sampler]") {
    UniformFrameSampler sampler;

    SECTION("Metadata preserved after sampling") {
        ClipContainer clip = createTestClip(10, "camera_123");
        clip.clip_id = "clip_456";

        sampler.sampleFrames(clip, 5);

        REQUIRE(clip.camera_id == "camera_123");
        REQUIRE(clip.clip_id == "clip_456");
        REQUIRE(clip.frames.size() == 10); 
    }
}
