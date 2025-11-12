#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "frame_batch.hpp"
#include "frame_extractor.hpp"
#include <opencv2/imgcodecs.hpp>

int main() {
    const std::filesystem::path input_dir = "saved_videos";
    const std::filesystem::path output_dir = "extracted_frames";
    std::filesystem::create_directories(output_dir);
    BlockingQueue<FrameBatch> queue;

    extract_frames_single(input_dir, queue);

    std::size_t batch_count = 0;
    std::size_t logged = 0;
    std::size_t saved_images = 0;

    while (auto batch_opt = queue.pop()) {
        const FrameBatch& batch = *batch_opt;
        ++batch_count;

        if (logged < 5) {
            std::cout << "帧 " << batch.frame_index << " 包含视角数: " << batch.frames.size()
                      << std::endl;
            ++logged;
        }

        for (const auto& [cam_id, frame] : batch.frames) {
            if (frame.empty()) {
                continue;
            }

            std::ostringstream frame_dir_ss;
            frame_dir_ss << "frame_" << std::setw(6) << std::setfill('0') << batch.frame_index;
            const auto frame_dir = output_dir / frame_dir_ss.str();
            std::filesystem::create_directories(frame_dir);

            std::ostringstream oss;
            oss << "cam_" << cam_id << ".png";
            const auto save_path = frame_dir / oss.str();
            if (cv::imwrite(save_path.string(), frame)) {
                ++saved_images;
            }
        }
    }

    std::cout << "共生成帧批次: " << batch_count << std::endl;
    std::cout << "共保存图片: " << saved_images << std::endl;
    return 0;
}


