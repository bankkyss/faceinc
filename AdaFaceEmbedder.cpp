#include "AdaFaceEmbedder.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace adaface {
namespace {
Ort::Env& global_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "adaface_embedder");
    return env;
}
}  // namespace

Embedder::Embedder(const std::string& model_path, const std::string& provider) {
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(3);
    if (provider == "CUDAExecutionProvider") {
#ifdef USE_CUDA
        options.AppendExecutionProvider_CUDA({});
#else
        throw std::runtime_error("CUDA provider requested but binary not built with USE_CUDA");
#endif
    } else {
        options.DisableMemPattern();
        options.SetIntraOpNumThreads(1);
    }

    session_ = std::make_unique<Ort::Session>(global_env(), model_path.c_str(), options);

    auto input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_alloc.get();

    const size_t output_count = session_->GetOutputCount();
    output_names_.resize(output_count);
    output_name_ptrs_.resize(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        auto name_alloc = session_->GetOutputNameAllocated(i, allocator_);
        output_names_[i] = name_alloc.get();
        output_name_ptrs_[i] = output_names_[i].c_str();
    }

    auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto shape = output_info.GetShape();
    if (shape.size() != 2) {
        throw std::runtime_error("AdaFace ONNX output must be a 2D tensor [1, feature_dim]");
    }
    feature_length_ = static_cast<int>(shape[1]);
    if (feature_length_ <= 0) {
        throw std::runtime_error("Invalid feature length inferred from AdaFace ONNX model");
    }
}

std::vector<float> Embedder::embed(const cv::Mat& aligned_face_bgr) const {
    if (aligned_face_bgr.empty()) {
        throw std::invalid_argument("Embedder received empty image");
    }
    if (aligned_face_bgr.type() != CV_8UC3) {
        throw std::invalid_argument("Embedder expects a BGR CV_8UC3 image");
    }

    cv::Mat input_blob = preprocess(aligned_face_bgr);

    const std::array<int64_t, 4> input_dims = {
        1,
        static_cast<int64_t>(input_blob.size[1]),
        static_cast<int64_t>(input_blob.size[2]),
        static_cast<int64_t>(input_blob.size[3])
    };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_blob.ptr<float>(),
                                                              input_blob.total(),
                                                              input_dims.data(),
                                                              input_dims.size());

    const char* input_names[] = {input_name_.c_str()};
    Ort::RunOptions run_options;
    auto outputs = session_->Run(run_options, input_names, &input_tensor, 1,
                                 output_name_ptrs_.data(), output_name_ptrs_.size());

    const float* feature_ptr = outputs.front().GetTensorData<float>();
    std::vector<float> feature(feature_ptr, feature_ptr + feature_length_);

    float norm = 0.0f;
    for (float v : feature) {
        norm += v * v;
    }
    norm = std::sqrt(std::max(norm, 1e-12f));
    for (float& v : feature) {
        v /= norm;
    }

    return feature;
}

cv::Mat Embedder::preprocess(const cv::Mat& aligned_face_bgr) const {
    cv::Size input_size(aligned_face_bgr.cols, aligned_face_bgr.rows);
    cv::Mat blob = cv::dnn::blobFromImage(aligned_face_bgr, 1.0 / 128.0, input_size,
                                          cv::Scalar(127.5, 127.5, 127.5), false);
    return blob;
}

}  // namespace adaface
