#include "components/media_processing/include/MediaProcessor.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    using namespace vision_analysis;

    MediaProcessorConfig config;
    config.stream_type = StreamHandlerType::RTSP;
    config.sampler_type = FrameSamplerType::UNIFORM;
    config.detector_type = ObjectDetectorType::YOLO;
    config.storage_type = StorageType::MILVUS_DB;
    config.frames_per_clip = 30;
    config.sampled_frames_count = 5;
    config.model_path = "models/yolo.onnx";

    MediaProcessor processor(config);

    std::cout << "Adding RTSP streams..." << std::endl;
    processor.addRTSPConnection("rtsp://camera1.example.com/stream", "camera_1");
    processor.addRTSPConnection("rtsp://camera2.example.com/stream", "camera_2");

    std::cout << "Starting media processor..." << std::endl;
    processor.start();

    std::cout << "Running for 30 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(30));

    std::cout << "Queue sizes:" << std::endl;
    std::cout << "  Clips: " << processor.getClipQueueSize() << std::endl;
    std::cout << "  Sampled Frames: " << processor.getSampledFramesQueueSize() << std::endl;
    std::cout << "  Tracked Objects: " << processor.getTrackedObjectsQueueSize() << std::endl;

    std::cout << "Stopping media processor..." << std::endl;
    processor.stop();

    std::cout << "Demo completed!" << std::endl;
    return 0;
}