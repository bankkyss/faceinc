#ifndef AI_FACE_REG_CORE_CPP_ADAFACE_EMBEDDER_HPP
#define AI_FACE_REG_CORE_CPP_ADAFACE_EMBEDDER_HPP

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace adaface {

class Embedder {
public:
    Embedder(const std::string& model_path, const std::string& provider = "CPUExecutionProvider");

    std::vector<float> embed(const cv::Mat& aligned_face_bgr) const;

    int feature_length() const { return feature_length_; }

private:
    cv::Mat preprocess(const cv::Mat& aligned_face_bgr) const;

    mutable Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<const char*> output_name_ptrs_;
    int feature_length_ = 0;
};

}  // namespace adaface

#endif  // AI_FACE_REG_CORE_CPP_ADAFACE_EMBEDDER_HPP
