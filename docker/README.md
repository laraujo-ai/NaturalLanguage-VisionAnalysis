# Docker Setup for NaturalLanguage-VisionAnalysis

This directory contains Docker configurations for building and running the NLVA system with all dependencies built from scratch.

## Overview

The Docker setup builds the following from source:
- **ONNX Runtime 1.16.0** with CUDA and TensorRT support (ARM64)
- **Milvus SDK C++** as a shared library
- Plus installs system dependencies: OpenCV, GStreamer, etc.

## Quick Start

### 1. Build and Run Everything

From the project root:

```bash
# Build and start all services (NLVA + Milvus)
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

This will:
1. Build the NLVA Docker image with all dependencies from scratch (~1-2 hours first time)
2. Start Milvus, etcd, and MinIO services
3. Start the NLVA application

### 2. Build Only (No Run)

```bash
docker-compose build nlva
```

### 3. Interactive Development

To enter the container without running the application:

```bash
docker-compose run --rm nlva /bin/bash
```

Then inside the container:
```bash
# Rebuild the project after code changes
cd /workspace/cpp_source
bash rebuild.sh

# Run the application
cd build
./StreamProcessing-NaturalLanguageRetrieval /workspace/config.json
```

## File Structure

```
docker/
├── Dockerfile.build      # Multi-stage build for all dependencies
├── docker-compose.yml    # DEPRECATED: Use root docker-compose.yml
├── .dockerignore         # Files to exclude from build context
└── README.md            # This file

Root:
├── docker-compose.yml    # Main orchestration file (USE THIS)
```

## Architecture Notes

### Multi-Stage Build

The Dockerfile uses a multi-stage build to:
1. **builder-base**: Install system dependencies
2. **eigen-builder**: Build Eigen from source
3. **json-builder**: Build nlohmann_json from source
4. **onnx-builder**: Build ONNX Runtime with CUDA/TensorRT
5. **milvus-builder**: Build Milvus SDK and dependencies (gRPC, protobuf, abseil)
6. **runtime**: Final slim image with only runtime dependencies

This keeps the final image size manageable while building everything from scratch.

### CMake Version Patching

ONNX Runtime build may encounter CMake version conflicts in dependencies. The Dockerfile includes a workaround that removes `cmake_minimum_required` from dependency CMakeLists and uses a wrapper script.

### Memory Management

The build creates an 8GB swap file to handle memory-intensive compilation (especially ONNX Runtime and Milvus SDK). Adjust `PARALLEL_JOBS` if you encounter OOM issues.

## Configuration

### Build Arguments

You can customize the build with these arguments:

```bash
docker-compose build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime \
  --build-arg PARALLEL_JOBS=4 \
  --build-arg ONNXRUNTIME_TAG=v1.16.0 \
  --build-arg MILVUS_SDK_TAG=v2.3.2 \
  nlva
```

### Environment Variables

Set in docker-compose.yml:
- `NVIDIA_VISIBLE_DEVICES`: GPU access (default: all)
- `LD_LIBRARY_PATH`: Library search paths
- `DOCKER_VOLUME_DIRECTORY`: Volume mount location for Milvus data

## Services

### nlva
- Main application container
- Processes RTSP streams and performs object detection/tracking
- Stores embeddings in Milvus
- Network: host mode (for RTSP stream access)

### standalone (Milvus)
- Vector database for storing embeddings
- Port: 19530 (gRPC API)
- Port: 9091 (metrics)

### etcd
- Milvus metadata storage
- Internal service (not exposed)

### minio
- Milvus object storage
- Port: 9000 (API)
- Port: 9001 (console)

## Volumes

The following directories are mounted for development:

```yaml
volumes:
  - ./cpp_source:/workspace/cpp_source    # Source code
  - ./config.json:/workspace/config.json  # Configuration
  - ./weights:/workspace/weights          # ONNX models
  - ./clips:/workspace/clips              # Video output
```

This allows you to:
- Edit code on host and rebuild inside container
- Update configs without rebuilding
- Access output clips from host

## Common Commands

```bash
# Build the image
docker-compose build nlva

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f nlva

# Stop all services
docker-compose down

# Rebuild after code changes (without rebuilding dependencies)
docker-compose exec nlva bash -c "cd /workspace/cpp_source && bash rebuild.sh"

# Enter running container
docker-compose exec nlva /bin/bash

# Clean everything (including volumes)
docker-compose down -v

# Rebuild from scratch (no cache)
docker-compose build --no-cache nlva
```

## Troubleshooting

### Build Fails with CMake Version Errors

The Dockerfile includes a patch that removes `cmake_minimum_required` from dependencies. If you still encounter issues:
1. Build once to populate `_deps`
2. Apply the patch manually
3. Rebuild

### Out of Memory During Build

Reduce `PARALLEL_JOBS`:
```yaml
build:
  args:
    PARALLEL_JOBS: 2  # Instead of 4
```

### Library Not Found at Runtime

Check `LD_LIBRARY_PATH`:
```bash
docker-compose exec nlva bash -c "echo \$LD_LIBRARY_PATH"
docker-compose exec nlva bash -c "ldd /workspace/cpp_source/build/StreamProcessing-NaturalLanguageRetrieval"
```

### NVIDIA Runtime Not Available

Ensure nvidia-docker2 is installed:
```bash
# On Jetson
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Network Issues (RTSP Streams)

The container uses `network_mode: host` to access local network RTSP streams. If you need different networking:
- Remove `network_mode: host`
- Add appropriate port mappings
- Configure firewall rules

## Platform Support

### ARM64 (Jetson)
Default configuration. Uses `l4t-tensorrt` base image.

### x86_64 (Desktop GPU)
Change BASE_IMAGE in docker-compose.yml:
```yaml
build:
  args:
    BASE_IMAGE: nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

## Production Deployment

For production:
1. Build the image once: `docker-compose build nlva`
2. Tag and push to registry: `docker tag nlva:latest your-registry/nlva:v1.0`
3. Update docker-compose.yml to use the registry image
4. Deploy with: `docker-compose up -d`

## Development Workflow

1. Make code changes on host
2. Rebuild inside container: `docker-compose exec nlva bash -c "cd /workspace/cpp_source && bash rebuild.sh"`
3. Run tests: `docker-compose exec nlva bash -c "cd /workspace/cpp_source/build && ctest"`
4. Restart application: `docker-compose restart nlva`

## License

Same as parent project.
