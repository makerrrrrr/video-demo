#include "frame_extractor.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
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
}  // namespace

// 单线程遍历输入目录下的所有视频，按帧生成批次并推送到队列。
// 读完或遇到错误后自动关闭队列。
void extract_frames_single(const std::filesystem::path& input_dir,
                           BlockingQueue<FrameBatch>& output_queue) {
    std::vector<VideoStream> streams;
    int next_cam_id = 0;

    if (!std::filesystem::exists(input_dir)) {
        std::cerr << "输入目录不存在: " << input_dir << std::endl;
        output_queue.close();
        return;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file() || !is_video_file(entry.path())) {
            continue;
        }
        int cam_id = parse_cam_id(entry.path(), next_cam_id++);
		// entyr.path()  L"saved_videos\\cam_0"	std::filesystem::path
		// std::make_unique创建一个智能指针，该指针管理一个cv::VideoCapture对象，entry.path().string()作为字符串传递给cv::VideoCapture对象，用于打开视频
	    auto capture = std::make_unique<cv::VideoCapture>(entry.path().string());
        if (!capture->isOpened()) {
            std::cerr << "无法打开视频: " << entry.path() << std::endl;
            continue;
        }
        const double fps = capture->get(cv::CAP_PROP_FPS);
        //将摄像头编号、视频路径、视频捕获器、帧率封装成VideoStream结构体，并添加到streams容器中
        streams.push_back({cam_id, entry.path(), std::move(capture), fps});
    }

    std::sort(streams.begin(), streams.end(),
              [](const VideoStream& lhs, const VideoStream& rhs) { return lhs.cam_id < rhs.cam_id; });

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
        ++frame_index;
    }

    output_queue.close();
}


