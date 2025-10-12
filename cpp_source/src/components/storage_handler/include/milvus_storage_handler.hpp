#pragma once

#include "../../../common/include/interfaces.hpp"
#include "../../../common/include/utils.hpp"
#include "milvus/MilvusClient.h"
#include "milvus/Status.h"
#include "milvus/types/Constants.h"
#include "milvus/utils/FP16.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

namespace nl_video_analysis
{
    class MilvusStorageHandler : public IStorageHandler {
    public:
        MilvusStorageHandler(const std::string& clip_storage_type,
                             const std::string& clip_storage_path,
                             const std::string& db_host,
                             const int db_port,
                             const std::string& db_user,
                             const std::string& db_password);

        ~MilvusStorageHandler() override;
        std::string saveClip(const ClipContainer& clip, std::map<int64_t, std::vector<std::vector<float>>>& embeddings_map) override;

        private:
        bool connectToDatabase();
        std::string saveClipToDisk(const ClipContainer& clip);

        std::string clip_storage_type_;
        std::string clip_storage_path_;
        std::string db_host_;
        int db_port_;
        std::string db_user_;
        std::string db_password_;

        std::shared_ptr<milvus::MilvusClient> db_client_;
        bool is_connected_;
    };
}
