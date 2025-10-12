# NaturalLanguage-VisionAnalysis

A video retrieval system that enables both natural language and image-based queries over video content from RTSP camera streams. The system consists of two main components: a Python API for querying the Milvus vector database, and a C++ stream processing pipeline that generates video clips and embeddings for detected objects.

## Overview

The system processes live RTSP camera streams, extracts meaningful visual information through object detection and tracking, and stores semantic embeddings that enable both natural language and image-based retrieval of video content.

### Architecture

**Stream Processing Pipeline (C++)**
- RTSP stream ingestion and clip generation
- Frame sampling and object detection
- Object tracking across frames
- Embedding generation and storage in Milvus

**Query API (Python)**
- FastAPI interface for natural language and image-based queries
- Milvus vector database integration
- Semantic search over stored embeddings

## Stream Processing Pipeline

The video processing pipeline performs the following operations:

- **Stream Ingestion**: Connects to RTSP camera servers and captures video streams
- **Clip Generation**: Segments streams into fixed-length clips for processing
- **Frame Sampling**: Uniformly samples frames from each clip for analysis
- **Object Detection & Tracking**: Detects and tracks objects across sampled frames
- **Embedding Generation**: Generates embeddings for each tracked object and average-pools them into a single mean vector per object
- **Storage**: Persists embeddings and metadata to the Milvus vector database

### Stream Handling

The pipeline supports two stream handler implementations:
- **GStreamer**: Hardware-accelerated RTSP decoding (NVIDIA)
- **OpenCV**: File-based input for offline testing

The GStreamer implementation builds a pipeline with an `appsink` element to extract frames from the main loop thread. Frames are accumulated into clips and queued for downstream processing.

### Pipeline Orchestration

The **VideoAnalysisEngine** component serves as the central orchestrator, managing the entire processing workflow through a multi-threaded architecture:

**Clip Processing Thread**
- Retrieves clips from individual stream handler queues
- Creates `ClipContainer` objects with camera metadata (camera_id, clip_id, timestamps)
- Performs frame sampling on clips
- Enqueues clips into a shared processing pool

This design ensures that clips retain all necessary metadata for database storage, regardless of their source stream.

**Object Processing Thread**
- Retrieves clips from the processing pool
- Runs object detection on sampled frames
- Performs object tracking to associate detections across frames
- Generates embeddings for tracked objects and attributes them to tracks
- Invokes the storage handler to persist clips to disk and save average-pooled embeddings to Milvus

### Dependencies

The stream processing pipeline requires:

- **ONNX Runtime**: Deep learning inference engine
- **Eigen**: Linear algebra library for C++
- **Milvus C++ SDK**: Vector database client for embedding storage
- **GStreamer**: Video stream processing framework
- **OpenCV**: Computer vision utilities
- nlohman json
- grpc protobuffs

### Deployment

Both the stream processing pipeline and the FastAPI query service are containerized using Docker. This approach provides:

- **Consistency**: Eliminates the need to build ONNX Runtime and Milvus from source on each deployment target
- **Portability**: Bundles all dependencies (Eigen, nlohmann/json, etc.) into a single deployable image
- **Reproducibility**: Ensures identical runtime environments across different machines

The Docker images can be distributed via Docker Hub for streamlined deployment.

## Future Improvements

Potential enhancements to the system include:

- **Component Decoupling**: Refactor components to be standalone entities coordinated by the orchestrator, rather than tightly coupled within it
- **Processing Pools**: Implement pools of detector and embedding generator instances to enable parallel clip processing and reduce latency
- **Hardware Considerations**: The pooling approach requires careful resource management on NVIDIA Jetson platforms due to limited compute capacity