#include "frame_extractor.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

namespace {
    // 视频流结构体
struct VideoStream {
    int cam_id = -1; // 摄像头ID
    std::filesystem::path path; // 视频路径
    std::unique_ptr<cv::VideoCapture> cap; // 视频捕获器
    double fps = 0.0; // 帧率
};
// 判断是否是视频文件
bool is_video_file(const std::filesystem::path& path) {
    const auto ext = path.extension().string();
    if (ext.empty()) {
        return false;
    }
    static const std::vector<std::string> kVideoExt = {
        ".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV"};
    return std::find(kVideoExt.begin(), kVideoExt.end(), ext) != kVideoExt.end();
}
// 解析摄像头ID
// 尝试从目录的父目录（如cam_0）提取数字，成功返回这个编号，不成功返回备用编号
int parse_cam_id(const std::filesystem::path& path, int fallback) {
	//path :"saved_videos\\cam_0\\video_clip.mp4"	const std::filesystem::path &
    //parent_path():获取当前路径的父目录
    //filename获取最后一段路径（目录或者文件名）
    const auto parent = path.parent_path().filename().string(); //cam_0,cam_1,cam_2
    if (parent.rfind("cam_", 0) == 0) { //从索引为0的位置开始向后寻找cam
        try {
            return std::stoi(parent.substr(4));  //获取摄像头编号
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}
//收集所有视频流
//input_dir:视频目录路径
//返回值:视频流集合
std::vector<VideoStream> collect_streams(const std::filesystem::path& input_dir) {
    std::vector<VideoStream> streams;
    int next_cam_id = 0;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file() || !is_video_file(entry.path())) {
            continue;
        }
        int cam_id = parse_cam_id(entry.path(), next_cam_id++);
		//entyr.path()  L"saved_videos\\cam_0"	std::filesystem::path
        auto capture = std::make_unique<cv::VideoCapture>(entry.path().string());
        if (!capture->isOpened()) {
            std::cerr << "无法打开视频: " << entry.path() << std::endl;
            continue;
        }
        //获取视频的帧率
        const double fps = capture->get(cv::CAP_PROP_FPS);
        //将摄像头编号、视频路径、视频捕获器、帧率封装成VideoStream结构体，并添加到streams容器中
        streams.push_back({cam_id, entry.path(), std::move(capture), fps});
    }

    //sort:先按cam_id进行排序
    std::sort(streams.begin(), streams.end(),
              [](const VideoStream& lhs, const VideoStream& rhs) { return lhs.cam_id < rhs.cam_id; });
    
    return streams;
}

// 多线程抽帧中间结构体，用于在读取线程和调度线程之前传递单个摄像头的帧数据
struct FramePacket {
    int cam_id = -1;
    int frame_index = -1;
    cv::Mat frame;
    bool eof = false;
};
//render_worker：每个摄像头一个线程，负责读取该视频的所有帧
//packet_queue:所有读取线程把帧数据包放入这个共享队列
void reader_worker(VideoStream stream,
                   BlockingQueue<FramePacket>& packet_queue,
                   std::atomic<bool>& stop_flag) {
    int frame_index = 0;
    cv::Mat frame;
    while (!stop_flag.load()) {
        if (!stream.cap->read(frame)) {
            break;
        }
        // 读到了帧，打包成FramePacket(摄像头ID,帧索引,帧数据,是否是最后一帧)，放入共享队列
        packet_queue.push({stream.cam_id, frame_index++, frame.clone(), false});
    }
    //读完了，将空帧打包成一帧，放入共享队列，标志读取线程结束
    packet_queue.push({stream.cam_id, frame_index, cv::Mat(), true});
}
}  // namespace
// 单线程遍历输入目录下的所有视频，按帧生成批次并推送到队列。
// 读完或遇到错误后自动关闭队列。
void extract_frames_single(const std::filesystem::path& input_dir,
                           BlockingQueue<FrameBatch>& output_queue) {
    if (!std::filesystem::exists(input_dir)) {
        std::cerr << "输入目录不存在: " << input_dir << std::endl;
        output_queue.close();
        return;
    }

    auto streams = collect_streams(input_dir);
    if (streams.empty()) {
        std::cerr << "目录中未找到可用视频: " << input_dir << std::endl;
        output_queue.close();
        return;
    }

    int frame_index = 0;
    bool stop = false;

    while (!stop) {
        FrameBatch batch;
        batch.frame_index = frame_index;
        if (!streams.empty()) {
            const double fps = streams.front().fps;
            batch.timestamp = fps > 0.0 ? frame_index / fps : 0.0;
        }

        for (auto& stream : streams) {
            cv::Mat frame;
            //对streams中的每一路VideoStream对象，读取一帧图像
            //read:从视频流中读取一帧图像
            if (!stream.cap->read(frame)) {
                stop = true;
                break;
            }
            batch.frames.emplace(stream.cam_id, std::move(frame));
        }

        if (stop) {
            break;
        }

        output_queue.push(std::move(batch));
        //帧号＋1，用于下一帧的读取
        ++frame_index;
    }

    output_queue.close();
}

// 多线程遍历输入目录下的视频，同步抽帧并推送到队列。
// 读完或遇到错误后自动关闭队列。
void extract_frames_parallel(const std::filesystem::path& input_dir,
                              BlockingQueue<FrameBatch>& output_queue) {
    if (!std::filesystem::exists(input_dir)) {
        std::cerr << "输入目录不存在: " << input_dir << std::endl;
        output_queue.close();
        return;
    }

    auto streams = collect_streams(input_dir);
    if (streams.empty()) {
        std::cerr << "目录中未找到可用视频: " << input_dir << std::endl;
        output_queue.close();
        return;
    }

    BlockingQueue<FramePacket> packet_queue;
    std::atomic<bool> stop_flag = false;
    std::vector<std::thread> workers;
    workers.reserve(streams.size());

    std::vector<double> fps_values;
    fps_values.reserve(streams.size());

    for (auto& stream : streams) {
        fps_values.push_back(stream.fps);
        workers.emplace_back(reader_worker,
                             std::move(stream),
                             std::ref(packet_queue),
                             std::ref(stop_flag));
    }
    streams.clear();

    const std::size_t total_cams = fps_values.size();
    if (total_cams == 0) {
        output_queue.close();
        return;
    }

    double reference_fps = 0.0;
    for (double fps : fps_values) {
        if (fps > 0.0) {
            reference_fps = fps;
            break;
        }
    }

    std::map<int, std::map<int, cv::Mat>> frame_buffer;
    std::size_t finished_cams = 0;
    int next_batch_index = 0;
    int first_eof_frame_index = -1;
    std::map<int, int> cam_eof_indices;

    // 辅助函数：尝试输出已收齐的帧批次
    auto try_output_complete_batches = [&]() {
        while (true) {
            if (first_eof_frame_index >= 0 && next_batch_index >= first_eof_frame_index) {
                break;
            }

            auto it = frame_buffer.find(next_batch_index);
            if (it == frame_buffer.end()) {
                break;
            }

            if (it->second.size() != total_cams) {
                break;
            }

            FrameBatch batch;
            batch.frame_index = next_batch_index;
            batch.timestamp = reference_fps > 0.0 ? static_cast<double>(next_batch_index) / reference_fps : 0.0;
            batch.frames = std::move(it->second);
            frame_buffer.erase(it);
            output_queue.push(std::move(batch));
            ++next_batch_index;
        }
    };

    while (finished_cams < total_cams) {
        auto packet_opt = packet_queue.pop();
        if (!packet_opt) {
            if (packet_queue.closed() && finished_cams < total_cams) {
                break;
            }
            continue;
        }

        FramePacket packet = std::move(*packet_opt);
        if (packet.eof) {
            cam_eof_indices[packet.cam_id] = packet.frame_index;
            ++finished_cams;

            if (first_eof_frame_index < 0) {
                first_eof_frame_index = packet.frame_index;
            } else {
                first_eof_frame_index = std::min(first_eof_frame_index, packet.frame_index);
            }
            //确保所有摄像头都读完后才设置停止标志
            if(finished_cams == total_cams) {
                stop_flag.store(true);
            }
            try_output_complete_batches();
            continue;
        }

        auto eof_it = cam_eof_indices.find(packet.cam_id);
        if (eof_it != cam_eof_indices.end() && packet.frame_index >= eof_it->second) {
            continue;
        }

        auto& frames_for_index = frame_buffer[packet.frame_index];
        frames_for_index.emplace(packet.cam_id, std::move(packet.frame));
        try_output_complete_batches();
    }

    stop_flag.store(true);
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    packet_queue.close();

    while (true) {
        auto packet_opt = packet_queue.pop();
        if (!packet_opt) {
            break;
        }

        FramePacket packet = std::move(*packet_opt);
        if (packet.eof) {
            continue;
        }

        auto eof_it = cam_eof_indices.find(packet.cam_id);
        if (eof_it != cam_eof_indices.end() && packet.frame_index >= eof_it->second) {
            continue;
        }

        auto& frames_for_index = frame_buffer[packet.frame_index];
        frames_for_index.emplace(packet.cam_id, std::move(packet.frame));
        try_output_complete_batches();
    }

    try_output_complete_batches();
    output_queue.close();
}