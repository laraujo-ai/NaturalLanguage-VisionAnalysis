#include "../include/milvus_storage_handler.hpp"
#include "logger.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace nl_video_analysis {

MilvusStorageHandler::MilvusStorageHandler(const std::string& clip_storage_type,
                                           const std::string& clip_storage_path,
                                           const std::string& db_host,
                                           const int db_port,
                                           const std::string& db_user,
                                           const std::string& db_password)
    : clip_storage_type_(clip_storage_type),
      clip_storage_path_(clip_storage_path),
      db_host_(db_host),
      db_port_(db_port),
      db_user_(db_user),
      db_password_(db_password),
      is_connected_(false) {

    if (clip_storage_type_ == "disk") {
        std::filesystem::path storage_path(clip_storage_path_);
        if (!std::filesystem::exists(storage_path)) {
            std::filesystem::create_directories(storage_path);
            LOG_INFO("[MilvusStorageHandler] Created storage directory: {}", clip_storage_path_);
        }
    }

    db_client_ = milvus::MilvusClient::Create();

    if (!connectToDatabase()) {
        LOG_WARN("[MilvusStorageHandler] Failed to connect to Milvus database on initialization");
    }
}

MilvusStorageHandler::~MilvusStorageHandler() {
    if (db_client_ && is_connected_) {
        db_client_->Disconnect();
        LOG_INFO("[MilvusStorageHandler] Disconnected from Milvus database");
    }
}

bool MilvusStorageHandler::connectToDatabase() {
    if (!db_client_) {
        LOG_ERROR("[MilvusStorageHandler] Database client not initialized");
        return false;
    }

    milvus::ConnectParam connect_param{db_host_, static_cast<uint16_t>(db_port_), db_user_, db_password_};
    auto status = db_client_->Connect(connect_param);

    if (status.IsOk()) {
        is_connected_ = true;
        LOG_INFO("[MilvusStorageHandler] Successfully connected to Milvus at {}:{}", db_host_, db_port_);

        std::vector<std::string> db_names;
        status = db_client_->ListDatabases(db_names);
        if (status.IsOk()) {
            std::string db_list;
            for (const auto& name : db_names) {
                db_list += name + " ";
            }
            LOG_INFO("[MilvusStorageHandler] Available databases: {}", db_list);
        }

        return true;
    } else {
        is_connected_ = false;
        LOG_ERROR("[MilvusStorageHandler] Failed to connect to Milvus: {}", status.Message());
        return false;
    }
}

std::string MilvusStorageHandler::saveClipToDisk(const ClipContainer& clip) {
    if (clip_storage_type_ != "disk") {
        return "";
    }

    std::filesystem::path camera_dir = std::filesystem::path(clip_storage_path_) / clip.camera_id;
    if (!std::filesystem::exists(camera_dir)) {
        std::filesystem::create_directories(camera_dir);
    }

    std::stringstream filename;
    filename << clip.clip_id << ".mp4";
    std::filesystem::path clip_file_path = camera_dir / filename.str();

    if (!clip.frames.empty()) {
        int frame_width = clip.frames[0].cols;
        int frame_height = clip.frames[0].rows;
        double fps = 30.0;

        cv::VideoWriter video_writer(
            clip_file_path.string(),
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            cv::Size(frame_width, frame_height)
        );

        if (!video_writer.isOpened()) {
            LOG_ERROR("[MilvusStorageHandler] Failed to open video writer for: {}", clip_file_path.string());
            return "";
        }

        for (const auto& frame : clip.frames) {
            video_writer.write(frame);
        }

        video_writer.release();
        LOG_INFO("[MilvusStorageHandler] Saved clip to disk: {}", clip_file_path.string());
        return clip_file_path.string();
    }

    return "";
}

// will need to pass the embeddings for the clip to this method as well
std::string MilvusStorageHandler::saveClip(const ClipContainer& clip, std::map<int64_t, std::vector<std::vector<float>>>& embeddings_map) {
    if (!is_connected_) {
        LOG_INFO("[MilvusStorageHandler] Not connected to database, attempting to reconnect...");
        if (!connectToDatabase()) {
            LOG_ERROR("[MilvusStorageHandler] Failed to reconnect to database");
        }
    }

    std::string clip_path = saveClipToDisk(clip);
    for(const auto& [tracklet_id, embeddings_list] : embeddings_map)
    {
        std::vector<float> resulting_embedding = nl_video_analysis::averageTrackEmbeddings(embeddings_list);
    }

    if (clip_path.empty()) {
        LOG_ERROR("[MilvusStorageHandler] Failed to save clip to disk");
        return "";
    }

    double duration = (clip.end_timestamp_ms - clip.start_timestamp_ms) / 1000.0;
    LOG_INFO("[MilvusStorageHandler] Processed clip: ID={}, Camera={}, Frames={}, Sampled={}, Start={}ms, End={}ms, Duration={:.2f}s, Path={}",
             clip.clip_id, clip.camera_id, clip.frames.size(), clip.sampled_frames.size(),
             clip.start_timestamp_ms, clip.end_timestamp_ms, duration, clip_path);

    // TODO: Future implementation - save embeddings and metadata to Milvus
    // This would include:
    // 1. Generate embeddings for the clip (via CLIP model)
    // 2. Insert embeddings into Milvus collection
    // 3. Store metadata (clip_path, camera_id, timestamps, etc.)

    return clip_path;
}

} // namespace nl_video_analysis
