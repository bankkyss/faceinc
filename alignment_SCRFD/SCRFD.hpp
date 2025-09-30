#ifndef AI_FACE_REG_CORE_ALIGNMENT_SCRFD_HPP
#define AI_FACE_REG_CORE_ALIGNMENT_SCRFD_HPP

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace scrfd {

struct Detection {
    cv::Rect2f bbox;
    float score;
    std::array<cv::Point2f, 5> keypoints;
    bool has_keypoints = false;
};

class SCRFD {
public:
    SCRFD(const std::string& model_path,
          const std::vector<cv::Point2f>& reference_points,
          cv::Size image_size = cv::Size(112, 112));

    void prepare(int ctx_id, float nms_threshold = -1.0f, cv::Size input_size = {});

    std::pair<std::vector<Detection>, std::vector<std::array<cv::Point2f, 5>>>
    detect(const cv::Mat& img, float threshold = 0.5f, cv::Size input_size = {},
           int max_num = 0, const std::string& metric = "default");

    static std::pair<const Detection*, const std::array<cv::Point2f, 5>*>
    pick_face(std::vector<Detection>& detections,
              std::vector<std::array<cv::Point2f, 5>>& keypoints,
              const std::string& policy = "largest");

    cv::Mat align(const cv::Mat& img_bgr, const std::array<cv::Point2f, 5>& landmarks) const;

private:
    struct ForwardResult {
        std::vector<float> scores;
        std::vector<cv::Rect2f> bboxes;
        std::vector<std::array<cv::Point2f, 5>> kpss;
    };

    void init_model_io();
    ForwardResult forward(const cv::Mat& img, float threshold);
    std::vector<int> nms(const std::vector<cv::Rect2f>& bboxes,
                         const std::vector<float>& scores) const;
    std::vector<cv::Point2f> build_anchor_centers(int height, int width, int stride) const;

    cv::Size image_size_;
    std::vector<cv::Point2f> reference_points_;
    float nms_threshold_ = 0.4f;

    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<const char*> output_name_ptrs_;

    std::vector<int64_t> input_shape_;
    cv::Size fixed_input_size_;
    bool use_kps_ = false;
    int fmc_ = 0;
    std::vector<int> feat_stride_fpn_;
    int num_anchors_ = 1;

    mutable std::unordered_map<std::string, std::vector<cv::Point2f>> center_cache_;
};

}  // namespace scrfd

#endif  // AI_FACE_REG_CORE_ALIGNMENT_SCRFD_HPP
