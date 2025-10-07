#include "../include/frame_samplers.hpp"
#include <iostream>

namespace nl_video_analysis {

void UniformFrameSampler::sampleFrames(ClipContainer& clip, int num_frames) {
    std::vector<cv::Mat> sampled;

    if (clip.frames.empty()) {
        return;
    }

    int total_frames = clip.frames.size();
    num_frames = std::min(num_frames, total_frames);

    if (num_frames == 1) {
        sampled.push_back(clip.frames[total_frames / 2].clone());
    } else {
        double step = static_cast<double>(total_frames - 1) / (num_frames - 1);
        for (int i = 0; i < num_frames; ++i) {
            int index = static_cast<int>(i * step);
            sampled.push_back(clip.frames[index].clone());
        }
    }
    clip.sampled_frames = std::move(sampled);
}
}