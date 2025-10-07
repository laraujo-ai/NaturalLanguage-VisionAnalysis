#include "components/media_processing/include/MediaProcessor.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    using namespace vision_analysis;

    // Configure ONNX models and vision analysis pipeline
    VisionAnalysisConfig vision_config;
    vision_config.yolo_model_path = "models/yolov8n.onnx";
    vision_config.clip_model_path = "models/clip_vision.onnx";
    vision_config.class_names_file = "models/coco.names";
    vision_config.num_threads = 4;
    vision_config.confidence_threshold = 0.5f;
    vision_config.nms_threshold = 0.4f;
    vision_config.yolo_input_width = 640;
    vision_config.yolo_input_height = 640;
    vision_config.clip_input_size = 224;
    vision_config.clip_embedding_size = 512;

    // Configure media processor
    MediaProcessorConfig config;
    config.vision_config = vision_config;
    config.sampler_type = FrameSamplerType::UNIFORM;
    config.storage_type = StorageType::MILVUS_DB;
    config.frames_per_clip = 30;
    config.sampled_frames_count = 5;
    config.max_connections = 5;
    config.queue_max_size = 50;
    config.storage_path = "./storage";

    MediaProcessor processor(config);

    std::cout << "=== ONNX-Based Vision Analysis Pipeline Demo ===" << std::endl;
    std::cout << "Using YOLO model: " << vision_config.yolo_model_path << std::endl;
    std::cout << "Using CLIP model: " << vision_config.clip_model_path << std::endl;

    // Add different types of sources
    std::cout << "\nAdding video sources..." << std::endl;

    // Add RTSP streams from CCTV cameras
    processor.addRTSPConnection("rtsp://192.168.1.100:554/stream1", "cctv_entrance");
    processor.addRTSPConnection("rtsp://192.168.1.101:554/stream1", "cctv_parking");

    // Add video files
    processor.addVideoFile("/path/to/video1.mp4", "recorded_video_1");
    processor.addVideoFile("/path/to/video2.avi", "recorded_video_2");

    // Auto-detect source type (recommended approach)
    processor.addSource("rtsp://192.168.1.102:554/stream1");  // Will auto-detect as RTSP
    processor.addSource("/path/to/video3.mkv");               // Will auto-detect as video file

    std::cout << "\nStarting vision analysis pipeline..." << std::endl;
    processor.start();

    // Monitor pipeline for 60 seconds
    std::cout << "Running pipeline for 60 seconds..." << std::endl;
    for (int i = 0; i < 60; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (i % 10 == 0) {
            std::cout << "\n--- Pipeline Status (t=" << i << "s) ---" << std::endl;
            std::cout << "Clips in queue: " << processor.getClipQueueSize() << std::endl;
            std::cout << "Sampled frames in queue: " << processor.getSampledFramesQueueSize() << std::endl;
            std::cout << "Tracked objects in queue: " << processor.getTrackedObjectsQueueSize() << std::endl;
        }
    }

    std::cout << "\n=== Final Pipeline Statistics ===" << std::endl;
    std::cout << "Final clips in queue: " << processor.getClipQueueSize() << std::endl;
    std::cout << "Final sampled frames in queue: " << processor.getSampledFramesQueueSize() << std::endl;
    std::cout << "Final tracked objects in queue: " << processor.getTrackedObjectsQueueSize() << std::endl;

    std::cout << "\nStopping vision analysis pipeline..." << std::endl;
    processor.stop();

    std::cout << "Demo completed!" << std::endl;
    return 0;
}