#include "clip_image_encoder.hpp"
#include "../../../common/include/benchmark.hpp"
#include <stdexcept>

namespace nl_video_analysis {

    CLIPImageEncoder::CLIPImageEncoder(const std::string& model_path, const int num_threads, bool is_fp16)
        : IBaseModel<const cv::Mat&, std::vector<float>>(model_path, num_threads),
          is_fp16_(is_fp16)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();

        if (input_shape_.size() != 4) {
            throw std::runtime_error("Expected 4D input tensor for CLIP image encoder");
        }
        target_size_ = static_cast<int>(input_shape_[2]);
        LOG_INFO("CLIPImageEncoder initialized with target size: {}x{}", target_size_, target_size_);
    }
    std::vector<float> CLIPImageEncoder::encode(const cv::Mat& iFrame)
    {
        return this->run(iFrame);        
    }

    std::vector<Ort::Value> CLIPImageEncoder::preprocess(const cv::Mat& input) {
        nl_video_analysis::ScopedTimer timer("clip_preprocess");

        cv::Mat img_rgb;
        cv::cvtColor(input, img_rgb, cv::COLOR_BGR2RGB);

        int h = img_rgb.rows;
        int w = img_rgb.cols;
        int new_h, new_w;

        if (h < w) {
            new_h = target_size_;
            new_w = static_cast<int>(target_size_ * static_cast<float>(w) / h);
        } else {
            new_h = static_cast<int>(target_size_ * static_cast<float>(h) / w);
            new_w = target_size_;
        }

        cv::Mat img_resized;
        cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

        h = img_resized.rows;
        w = img_resized.cols;
        int top = (h - target_size_) / 2;
        int left = (w - target_size_) / 2;

        cv::Rect crop_region(left, top, target_size_, target_size_);
        cv::Mat img_cropped = img_resized(crop_region);

        cv::Mat img_float;
        img_cropped.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(img_float, channels);

        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean_[i]) / std_[i];
        }

        cv::Mat img_normalized;
        cv::merge(channels, img_normalized);

        std::vector<float> input_tensor_values(1 * 3 * target_size_ * target_size_);

        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < target_size_; ++h) {
                for (int w = 0; w < target_size_; ++w) {
                    int tensor_idx = c * target_size_ * target_size_ + h * target_size_ + w;
                    input_tensor_values[tensor_idx] = channels[c].at<float>(h, w);
                }
            }
        }

        std::vector<int64_t> input_shape = {1, 3, target_size_, target_size_};
        std::vector<Ort::Value> tensors;

        if (is_fp16_) {
            std::vector<Ort::Float16_t> input_data_fp16(input_tensor_values.size());
            for (size_t i = 0; i < input_tensor_values.size(); ++i) {
                input_data_fp16[i] = Ort::Float16_t(input_tensor_values[i]);
            }
            input_data_fp16_ = std::move(input_data_fp16);

            auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                memory_info_,
                input_data_fp16_.data(),
                input_data_fp16_.size(),
                input_shape.data(),
                input_shape.size()
            );
            tensors.push_back(std::move(tensor));
        } else {
            input_data_fp32_ = std::move(input_tensor_values);

            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info_,
                input_data_fp32_.data(),
                input_data_fp32_.size(),
                input_shape.data(),
                input_shape.size()
            );
            tensors.push_back(std::move(tensor));
        }

        return tensors;
    }

    std::vector<float> CLIPImageEncoder::postprocess(std::vector<Ort::Value>& output_tensors) {
        nl_video_analysis::ScopedTimer timer("clip_postprocess");

        if (output_tensors.empty()) {
            throw std::runtime_error("No output tensors from CLIP model");
        }

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();

        size_t embedding_size = 1;
        for (auto dim : shape) {
            embedding_size *= dim;
        }

        std::vector<float> embedding(output_data, output_data + embedding_size);

        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-6f) {
            for (float& val : embedding) {
                val /= norm;
            }
        }

        return embedding;
    }

}