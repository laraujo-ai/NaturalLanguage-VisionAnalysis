#ifndef CLIP_IMAGE_ENCODER_HPP
#define CLIP_IMAGE_ENCODER_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "../../../common/include/base_model.hpp"

namespace nl_vision_analysis {

    class CLIPImageEncoder : public IBaseModel<const cv::Mat&, std::vector<float>> {
        public:
            CLIPImageEncoder(const std::string& model_path, const int num_threads, bool is_fp16 = false);
            std::vector<float> encode(const cv::Mat& iFrame);
            ~CLIPImageEncoder() override = default;

        protected:
            std::vector<Ort::Value> preprocess(const cv::Mat& input) override;
            std::vector<float> postprocess(std::vector<Ort::Value>& output_tensors) override;

        private:
            bool is_fp16_;
            int target_size_ = 224;

            const std::vector<float> mean_ = {0.48145466f, 0.4578275f, 0.40821073f};
            const std::vector<float> std_ = {0.26862954f, 0.26130258f, 0.27577711f};

            std::vector<int64_t> input_shape_;

            // Member variables to persist tensor data (similar to YOLOXDetector)
            std::vector<Ort::Float16_t> input_data_fp16_;
            std::vector<float> input_data_fp32_;
    };
}

#endif // CLIP_IMAGE_ENCODER_HPP
