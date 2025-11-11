#include <filesystem>
#include <iostream>

#include "frame_batch.hpp"
#include "frame_extractor.hpp"

int main() {
    const std::filesystem::path input_dir = "saved_videos";
    BlockingQueue<FrameBatch> queue;

    extract_frames_single(input_dir, queue);

    std::size_t batch_count = 0;
    std::size_t logged = 0;

    while (auto batch_opt = queue.pop()) {
        const FrameBatch& batch = *batch_opt;
        ++batch_count;

        if (logged < 5) {
            std::cout << "帧 " << batch.frame_index << " 包含视角数: " << batch.frames.size()
                      << std::endl;
            ++logged;
        }
    }

    std::cout << "共生成帧批次: " << batch_count << std::endl;
    return 0;
}


