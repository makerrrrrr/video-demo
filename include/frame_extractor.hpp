#pragma once

#include <filesystem>

#include "frame_batch.hpp"

// 单线程遍历输入目录下的所有视频，按帧生成批次并推送到队列。
// 读完或遇到错误后自动关闭队列。
void extract_frames_single(const std::filesystem::path& input_dir,
                           BlockingQueue<FrameBatch>& output_queue);



