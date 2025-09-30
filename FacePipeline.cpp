#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "AdaFaceEmbedder.hpp"
#include "alignment_SCRFD/SCRFD.hpp"

namespace {
std::vector<cv::Point2f> reference_points_default_square_112() {
    return {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    };
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Feature vectors must have equal length");
    }
    float dot = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

void visualize(const cv::Mat& img1, const cv::Mat& aligned1,
               const cv::Mat& img2, const cv::Mat& aligned2,
               float similarity, float threshold, const std::string& window_title) {
    const cv::Size display_size(224, 224);
    cv::Mat img1_display, img2_display, aligned1_display, aligned2_display;
    cv::resize(img1, img1_display, display_size);
    cv::resize(img2, img2_display, display_size);
    cv::resize(aligned1, aligned1_display, display_size);
    cv::resize(aligned2, aligned2_display, display_size);

    cv::Mat top, bottom, combined;
    cv::hconcat(img1_display, aligned1_display, top);
    cv::hconcat(img2_display, aligned2_display, bottom);
    cv::vconcat(top, bottom, combined);

    std::ostringstream ss;
    ss << "Similarity: " << std::fixed << std::setprecision(4) << similarity;
    cv::putText(combined, ss.str(), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);

    const char* label = (similarity >= threshold) ? "Same Person" : "Different Person";
    cv::putText(combined, label, {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 0, 255), 2);

    cv::imshow(window_title, combined);
    std::cout << "Press any key on the visualization window to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyWindow(window_title);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <adaface.onnx> <scrfd.onnx> <image_a> <image_b> [threshold]" << std::endl;
        return 1;
    }

    const std::string adaface_model = argv[1];
    const std::string scrfd_model = argv[2];
    const std::string image_path_a = argv[3];
    const std::string image_path_b = argv[4];
    const float threshold = (argc > 5) ? std::stof(argv[5]) : 0.4f;

    try {
        const auto reference_points = reference_points_default_square_112();
        scrfd::SCRFD detector(scrfd_model, reference_points, cv::Size(112, 112));
        detector.prepare(-1, 0.4f, cv::Size(480, 480));

        adaface::Embedder embedder(adaface_model);

        cv::Mat img_a = cv::imread(image_path_a);
        cv::Mat img_b = cv::imread(image_path_b);
        if (img_a.empty() || img_b.empty()) {
            throw std::runtime_error("Failed to load one or both images");
        }

        auto process_image = [&](const cv::Mat& img) {
            auto [detections, keypoints] = detector.detect(img, 0.3f, cv::Size(480, 480));
            if (detections.empty()) {
                std::cout << "WARNING: no face detected, returning resized image" << std::endl;
                cv::Mat resized;
                cv::resize(img, resized, cv::Size(112, 112));
                return std::make_pair(resized, std::vector<float>());
            }

            auto selection = scrfd::SCRFD::pick_face(detections, keypoints, "largest");
            const auto* det_ptr = selection.first;
            const auto* kp_ptr = selection.second;

            cv::Mat aligned;
            if (kp_ptr != nullptr) {
                aligned = detector.align(img, *kp_ptr);
            } else {
                cv::Mat resized;
                cv::resize(img, resized, cv::Size(112, 112));
                aligned = resized;
            }

            std::vector<float> feature = embedder.embed(aligned);
            return std::make_pair(aligned, feature);
        };

        auto [aligned_a, feat_a] = process_image(img_a);
        auto [aligned_b, feat_b] = process_image(img_b);

        if (feat_a.empty() || feat_b.empty()) {
            throw std::runtime_error("Unable to extract features for comparison");
        }

        const float similarity = cosine_similarity(feat_a, feat_b);
        std::cout << "Cosine Similarity: " << std::fixed << std::setprecision(4)
                  << similarity << std::endl;
        if (similarity >= threshold) {
            std::cout << "Result: Same Person (>= " << threshold << ")" << std::endl;
        } else {
            std::cout << "Result: Different Person (< " << threshold << ")" << std::endl;
        }

        visualize(img_a, aligned_a, img_b, aligned_b, similarity, threshold,
                  "Face Comparison Result");
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
