#include "SCRFD.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace scrfd {
namespace {
Ort::Env& global_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "scrfd");
    return env;
}

std::string cache_key(int height, int width, int stride) {
    std::ostringstream oss;
    oss << height << "x" << width << "@" << stride;
    return oss.str();
}

}  // namespace

SCRFD::SCRFD(const std::string& model_path,
             const std::vector<cv::Point2f>& reference_points,
             cv::Size image_size)
    : image_size_(image_size), reference_points_(reference_points) {
    if (reference_points_.size() != 5) {
        throw std::invalid_argument("SCRFD expects exactly five reference facial points");
    }

    Ort::SessionOptions options;
    options.SetLogSeverityLevel(3);
    session_ = std::make_unique<Ort::Session>(global_env(), model_path.c_str(), options);
    init_model_io();
}

void SCRFD::prepare(int ctx_id, float nms_threshold, cv::Size input_size) {
    if (ctx_id >= 0) {
        // GPU selection happens when creating the session; we expose the signature for parity.
    }

    if (nms_threshold >= 0.0f) {
        nms_threshold_ = nms_threshold;
    }

    if (input_size.width > 0 && input_size.height > 0) {
        fixed_input_size_ = input_size;
    }
}

std::pair<std::vector<Detection>, std::vector<std::array<cv::Point2f, 5>>>
SCRFD::detect(const cv::Mat& img, float threshold, cv::Size input_size,
              int max_num, const std::string& metric) {
    cv::Size target_size = input_size;
    if (target_size.width == 0 || target_size.height == 0) {
        if (fixed_input_size_.width == 0 || fixed_input_size_.height == 0) {
            throw std::runtime_error("SCRFD input size is undefined; call prepare() or use a fixed-size model");
        }
        target_size = fixed_input_size_;
    }

    const float im_ratio = static_cast<float>(img.rows) / static_cast<float>(img.cols);
    const float model_ratio = static_cast<float>(target_size.height) / static_cast<float>(target_size.width);

    int new_height = 0;
    int new_width = 0;
    if (im_ratio > model_ratio) {
        new_height = target_size.height;
        new_width = static_cast<int>(std::round(new_height / im_ratio));
    } else {
        new_width = target_size.width;
        new_height = static_cast<int>(std::round(new_width * im_ratio));
    }

    const float det_scale = static_cast<float>(new_height) / static_cast<float>(img.rows);
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_width, new_height));

    cv::Mat det_img(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
    resized_img.copyTo(det_img(cv::Rect(0, 0, new_width, new_height)));

    auto forward_result = forward(det_img, threshold);

    auto& scores = forward_result.scores;
    auto& bboxes = forward_result.bboxes;
    auto& kpss = forward_result.kpss;

    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

    std::vector<cv::Rect2f> ordered_bboxes(order.size());
    std::vector<float> ordered_scores(order.size());
    std::vector<std::array<cv::Point2f, 5>> ordered_kpss;
    if (use_kps_) {
        ordered_kpss.resize(order.size());
    }

    for (std::size_t i = 0; i < order.size(); ++i) {
        const int idx = order[i];
        ordered_bboxes[i] = bboxes[idx];
        ordered_scores[i] = scores[idx];
        if (use_kps_) {
            ordered_kpss[i] = kpss[idx];
        }
    }

    auto keep = nms(ordered_bboxes, ordered_scores);

    std::vector<Detection> detections;
    std::vector<std::array<cv::Point2f, 5>> keypoints_out;
    detections.reserve(keep.size());
    if (use_kps_) {
        keypoints_out.reserve(keep.size());
    }

    for (int idx : keep) {
        cv::Rect2f bbox = ordered_bboxes[idx];
        bbox.x /= det_scale;
        bbox.y /= det_scale;
        bbox.width /= det_scale;
        bbox.height /= det_scale;

        Detection det;
        det.bbox = bbox;
        det.score = ordered_scores[idx];
        det.has_keypoints = use_kps_;
        if (use_kps_) {
            std::array<cv::Point2f, 5> kp = ordered_kpss[idx];
            for (auto& p : kp) {
                p.x /= det_scale;
                p.y /= det_scale;
            }
            det.keypoints = kp;
            keypoints_out.push_back(kp);
        }
        detections.push_back(det);
    }

    if (max_num > 0 && detections.size() > static_cast<std::size_t>(max_num)) {
        std::vector<float> values(detections.size());
        const cv::Point2f img_center(static_cast<float>(img.cols) / 2.0f,
                                     static_cast<float>(img.rows) / 2.0f);
        for (std::size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
        const float area = (det.bbox.width + 1.0f) * (det.bbox.height + 1.0f);
            const cv::Point2f center(det.bbox.x + det.bbox.width / 2.0f,
                                     det.bbox.y + det.bbox.height / 2.0f);
            const cv::Point2f diff = center - img_center;
            const float offset_dist_squared = diff.dot(diff);
            values[i] = (metric == "max") ? area : (area - offset_dist_squared * 2.0f);
        }
        std::vector<int> idx(values.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return values[a] > values[b]; });
        idx.resize(max_num);

        std::vector<Detection> filtered;
        std::vector<std::array<cv::Point2f, 5>> filtered_kps;
        filtered.reserve(idx.size());
        if (use_kps_) {
            filtered_kps.reserve(idx.size());
        }
        for (int i : idx) {
            filtered.push_back(detections[i]);
            if (use_kps_) {
                filtered_kps.push_back(keypoints_out[i]);
            }
        }
        detections.swap(filtered);
        if (use_kps_) {
            keypoints_out.swap(filtered_kps);
        }
    }

    return {detections, keypoints_out};
}

std::pair<const Detection*, const std::array<cv::Point2f, 5>*>
SCRFD::pick_face(std::vector<Detection>& detections,
                 std::vector<std::array<cv::Point2f, 5>>& keypoints,
                 const std::string& policy) {
    if (detections.empty()) {
        return {nullptr, nullptr};
    }

    int idx = 0;
    if (policy == "largest") {
        float best_area = 0.0f;
        for (std::size_t i = 0; i < detections.size(); ++i) {
            const auto& bbox = detections[i].bbox;
            const float area = bbox.width * bbox.height;
            if (area > best_area) {
                best_area = area;
                idx = static_cast<int>(i);
            }
        }
    } else if (policy == "highest_score") {
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].score > best_score) {
                best_score = detections[i].score;
                idx = static_cast<int>(i);
            }
        }
    }

    const Detection* det_ptr = &detections[idx];
    const std::array<cv::Point2f, 5>* kp_ptr = nullptr;
    if (!keypoints.empty()) {
        kp_ptr = &keypoints[idx];
    }
    return {det_ptr, kp_ptr};
}

cv::Mat SCRFD::align(const cv::Mat& img_bgr, const std::array<cv::Point2f, 5>& landmarks) const {
    cv::Mat src_img;
    if (img_bgr.channels() == 1) {
        cv::cvtColor(img_bgr, src_img, cv::COLOR_GRAY2BGR);
    } else {
        src_img = img_bgr;
    }

    std::vector<cv::Point2f> lmk(landmarks.begin(), landmarks.end());
    cv::Mat matrix = cv::estimateAffinePartial2D(lmk, reference_points_);
    if (matrix.empty()) {
        throw std::runtime_error("Failed to estimate similarity transform for alignment");
    }

    cv::Mat aligned;
    cv::warpAffine(src_img, aligned, matrix, image_size_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return aligned;
}

void SCRFD::init_model_io() {
    auto input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_alloc.get();

    auto input_type_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    input_shape_ = input_type_info.GetShape();
    if (input_shape_.size() >= 4) {
        const int64_t height = input_shape_[2];
        const int64_t width = input_shape_[3];
        if (height > 0 && width > 0) {
            fixed_input_size_ = cv::Size(static_cast<int>(width), static_cast<int>(height));
        }
    }

    const size_t output_count = session_->GetOutputCount();
    output_names_.resize(output_count);
    output_name_ptrs_.resize(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        auto name_alloc = session_->GetOutputNameAllocated(i, allocator_);
        output_names_[i] = name_alloc.get();
        output_name_ptrs_[i] = output_names_[i].c_str();
    }

    if (output_count == 6 || output_count == 9) {
        fmc_ = 3;
        feat_stride_fpn_ = {8, 16, 32};
        num_anchors_ = 2;
        use_kps_ = (output_count == 9);
    } else if (output_count == 10 || output_count == 15) {
        fmc_ = 5;
        feat_stride_fpn_ = {8, 16, 32, 64, 128};
        num_anchors_ = 1;
        use_kps_ = (output_count == 15);
    } else {
        throw std::runtime_error("Unsupported SCRFD ONNX output configuration");
    }
}

SCRFD::ForwardResult SCRFD::forward(const cv::Mat& img, float threshold) {
    const cv::Size input_size(img.cols, img.rows);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 128.0, input_size, cv::Scalar(127.5, 127.5, 127.5), true);

    const std::array<int64_t, 4> input_dims = {
        1,
        static_cast<int64_t>(blob.size[1]),
        static_cast<int64_t>(blob.size[2]),
        static_cast<int64_t>(blob.size[3])
    };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(),
                                                              blob.total(), input_dims.data(),
                                                              input_dims.size());

    const char* input_names[] = {input_name_.c_str()};
    Ort::RunOptions run_options;
    auto output_tensors = session_->Run(run_options, input_names, &input_tensor, 1,
                                        output_name_ptrs_.data(), output_name_ptrs_.size());

    ForwardResult result;

    const int input_height = static_cast<int>(blob.size[2]);
    const int input_width = static_cast<int>(blob.size[3]);

    for (int idx = 0; idx < fmc_; ++idx) {
        const int stride = feat_stride_fpn_[idx];
        const auto& score_tensor = output_tensors.at(idx);
        const auto& bbox_tensor = output_tensors.at(idx + fmc_);

        auto score_info = score_tensor.GetTensorTypeAndShapeInfo();
        auto bbox_info = bbox_tensor.GetTensorTypeAndShapeInfo();
        const auto score_shape = score_info.GetShape();
        const auto bbox_shape = bbox_info.GetShape();

        const int height = static_cast<int>(score_shape[2]);
        const int width = static_cast<int>(score_shape[3]);
        const int locations = height * width;

        const float* score_data = score_tensor.GetTensorData<float>();
        const float* bbox_data = bbox_tensor.GetTensorData<float>();

        const int anchors_per_loc = num_anchors_;
        const int score_channels = static_cast<int>(score_shape[1]);
        const int cls_per_anchor = score_channels / anchors_per_loc;

        std::vector<float> scores_curr;
        scores_curr.reserve(locations * anchors_per_loc);

        for (int loc = 0; loc < locations; ++loc) {
            for (int anchor = 0; anchor < anchors_per_loc; ++anchor) {
                const int channel_offset = anchor * cls_per_anchor * locations;
                const float* anchor_ptr = score_data + channel_offset;
                const float* positive_ptr = (cls_per_anchor == 2)
                                                ? anchor_ptr + locations
                                                : anchor_ptr;
                scores_curr.push_back(positive_ptr[loc]);
            }
        }

        const int bbox_channels = static_cast<int>(bbox_shape[1]);
        const int bbox_per_anchor = bbox_channels / anchors_per_loc;
        if (bbox_per_anchor != 4) {
            throw std::runtime_error("SCRFD expects 4 bbox channels per anchor");
        }

        std::vector<cv::Vec4f> bbox_deltas;
        bbox_deltas.reserve(locations * anchors_per_loc);
        for (int loc = 0; loc < locations; ++loc) {
            for (int anchor = 0; anchor < anchors_per_loc; ++anchor) {
                const float* anchor_ptr = bbox_data + anchor * bbox_per_anchor * locations;
                const float* left = anchor_ptr;
                const float* top = anchor_ptr + locations;
                const float* right = anchor_ptr + 2 * locations;
                const float* bottom = anchor_ptr + 3 * locations;
                bbox_deltas.emplace_back(left[loc] * stride, top[loc] * stride,
                                         right[loc] * stride, bottom[loc] * stride);
            }
        }

        std::vector<std::array<cv::Point2f, 5>> kps_curr;
        if (use_kps_) {
            const auto& kps_tensor = output_tensors.at(idx + fmc_ * 2);
            auto kps_info = kps_tensor.GetTensorTypeAndShapeInfo();
            const auto kps_shape = kps_info.GetShape();
            const float* kps_data = kps_tensor.GetTensorData<float>();
            const int kps_channels = static_cast<int>(kps_shape[1]);
            const int coords_per_anchor = kps_channels / anchors_per_loc;
            if (coords_per_anchor != 10) {
                throw std::runtime_error("SCRFD expects 10 keypoint channels per anchor");
            }
            kps_curr.reserve(locations * anchors_per_loc);
            for (int loc = 0; loc < locations; ++loc) {
                for (int anchor = 0; anchor < anchors_per_loc; ++anchor) {
                    const float* anchor_ptr = kps_data + anchor * coords_per_anchor * locations;
                    std::array<cv::Point2f, 5> kp{};
                    for (int k = 0; k < 5; ++k) {
                        const float px = anchor_ptr[k * 2 * locations + loc] * stride;
                        const float py = anchor_ptr[(k * 2 + 1) * locations + loc] * stride;
                        kp[k] = cv::Point2f(px, py);
                    }
                    kps_curr.push_back(kp);
                }
            }
        }

        const std::string key = cache_key(height, width, stride);
        auto cache_it = center_cache_.find(key);
        std::vector<cv::Point2f> centers;
        if (cache_it != center_cache_.end()) {
            centers = cache_it->second;
        } else {
            centers = build_anchor_centers(height, width, stride);
            if (center_cache_.size() < 100) {
                center_cache_.emplace(key, centers);
            }
        }

        for (std::size_t i = 0; i < scores_curr.size(); ++i) {
            if (scores_curr[i] < threshold) {
                continue;
            }
            const cv::Point2f& center = centers[i];
            const cv::Vec4f& offset = bbox_deltas[i];
            const float x1 = center.x - offset[0];
            const float y1 = center.y - offset[1];
            const float x2 = center.x + offset[2];
            const float y2 = center.y + offset[3];

            result.scores.push_back(scores_curr[i]);
            result.bboxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            if (use_kps_) {
                std::array<cv::Point2f, 5> kp = kps_curr[i];
                for (auto& p : kp) {
                    p.x += center.x;
                    p.y += center.y;
                }
                result.kpss.push_back(kp);
            }
        }
    }

    return result;
}

std::vector<int> SCRFD::nms(const std::vector<cv::Rect2f>& bboxes,
                            const std::vector<float>& scores) const {
    std::vector<int> order(bboxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(bboxes.size(), false);

    for (std::size_t _i = 0; _i < order.size(); ++_i) {
        const int i = order[_i];
        if (suppressed[i]) {
            continue;
        }
        keep.push_back(i);
        const auto& bbox_i = bboxes[i];
        const float area_i = (bbox_i.width + 1.0f) * (bbox_i.height + 1.0f);

        for (std::size_t _j = _i + 1; _j < order.size(); ++_j) {
            const int j = order[_j];
            if (suppressed[j]) {
                continue;
            }
            const auto& bbox_j = bboxes[j];
            const float xx1 = std::max(bbox_i.x, bbox_j.x);
            const float yy1 = std::max(bbox_i.y, bbox_j.y);
            const float xx2 = std::min(bbox_i.x + bbox_i.width, bbox_j.x + bbox_j.width);
            const float yy2 = std::min(bbox_i.y + bbox_i.height, bbox_j.y + bbox_j.height);

            const float w = std::max(0.0f, xx2 - xx1 + 1.0f);
            const float h = std::max(0.0f, yy2 - yy1 + 1.0f);
            const float inter = w * h;
            const float area_j = (bbox_j.width + 1.0f) * (bbox_j.height + 1.0f);
            const float ovr = inter / (area_i + area_j - inter + 1e-12f);
            if (ovr > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

std::vector<cv::Point2f> SCRFD::build_anchor_centers(int height, int width, int stride) const {
    std::vector<cv::Point2f> centers;
    centers.reserve(height * width * num_anchors_);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Point2f base(static_cast<float>(x * stride), static_cast<float>(y * stride));
            for (int n = 0; n < num_anchors_; ++n) {
                centers.push_back(base);
            }
        }
    }
    return centers;
}

}  // namespace scrfd
