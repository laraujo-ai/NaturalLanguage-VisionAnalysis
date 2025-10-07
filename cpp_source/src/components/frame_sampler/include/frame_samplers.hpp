#pragma once

#include "../../../common/include/interfaces.hpp"
#include "/home/nvidia/projects/NaturalLanguage-VisionAnalysis/cpp_source/src/components/stream_handler/include/vision_stream_handlers.hpp"
#include <algorithm>
#include <random>

namespace nl_video_analysis {

class UniformFrameSampler : public IFrameSampler {
public:
    ClipContainer() = default;
    void sampleFrames(const ClipContainer& clip, int num_frames) override;
};
}