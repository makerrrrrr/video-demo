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
        packet_queue.push({stream.cam_id, frame_index++, frame, false});
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
    //1、收集视频流
    auto streams = collect_streams(input_dir);
    if (streams.empty()) {
        std::cerr << "目录中未找到可用视频: " << input_dir << std::endl;
        output_queue.close();
        return;
    }
    //2、创建共享资源
    BlockingQueue<FramePacket> packet_queue; //所有读取线程都往这里放帧数据包
    std::atomic<bool> stop_flag = false; //停止信号，所有线程都用这个标志来决定是否停止

    std::vector<std::thread> workers; //创建线程数组
    workers.reserve(streams.size()); //预分配容量为streams.size()，减少后续添加线程时的重分配

    std::vector<double> fps_values; //创建帧率数组
    fps_values.reserve(streams.size()); //预分配帧率容量

    //当workers.emplace_back执行时，会创建一个std::thread对象并立即启动线程，开始执行reader_worker函数
    for (auto& stream : streams) {
        fps_values.push_back(stream.fps);
        workers.emplace_back(reader_worker,//要执行的函数
                             std::move(stream),
                             //参数1，std::move(stream)将stream移动到新线程，VideoStream 包含 std::unique_ptr<cv::VideoCapture>，不可复制，只能移动,移动后原stream不可再用
                             std::ref(packet_queue),//参数2，传递引用，多线程共享同一队列
                             std::ref(stop_flag));//参数3，传递引用，多线程共享同一停止标注
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

    //3、创建帧缓冲区
    //frame_buffer[帧索引][摄像头ID] = 图像
    std::map<int, std::map<int, cv::Mat>> frame_buffer;
    std::size_t finished_cams = 0; //已读完的摄像头数量
    int next_batch_index = 0; //下一帧的索引
    int first_eof_frame_index = -1; //第一个EOF的帧索引（模拟单线程行为：第一个摄像头读完就停止）
    
    // 辅助函数：尝试输出已收齐的帧批次
    auto try_output_complete_batches = [&]() {
        while (true) {
            // 如果已收到第一个EOF，且当前帧索引 >= EOF的帧索引，说明无法再输出更多批次了
            // EOF的frame_index是下一个应该读取的帧索引，实际最后一个有效帧是frame_index-1
            // 所以当next_batch_index >= first_eof_frame_index时，说明已经处理完所有有效帧
            if (first_eof_frame_index >= 0 && next_batch_index >= first_eof_frame_index) {
                break;
            }
            
            auto it = frame_buffer.find(next_batch_index);
            if (it == frame_buffer.end()) {
                // 这个帧索引的批次还没有收齐，无法继续输出后续批次
                break;
            }
            
            // 检查这个批次是否完整（所有摄像头都有这一帧）
            if (it->second.size() != total_cams) {
                // 批次不完整，无法输出，停止检查后续批次
                break;
            }
            
            //当某个帧号收齐所有摄像头时，组装成FrameBatch对象，推送到输出队列
            FrameBatch batch;
            batch.frame_index = next_batch_index;
            batch.timestamp = reference_fps > 0.0 ? static_cast<double>(next_batch_index) / reference_fps : 0.0;
            batch.frames = std::move(it->second);
            frame_buffer.erase(it);
            output_queue.push(std::move(batch));
            ++next_batch_index;
        }
    };
    
    //4、主循环：不断从队列取数据包
    while (finished_cams < total_cams) {
        auto packet_opt = packet_queue.pop(); //从队列中取一个数据包
        if (!packet_opt) {
            // 如果队列已关闭且没有更多数据包，但还有摄像头未完成，继续等待
            if (packet_queue.closed() && finished_cams < total_cams) {
                // 队列已关闭但还有摄像头未完成，可能是异常情况
                break;
            }
            continue;
        }

        FramePacket packet = std::move(*packet_opt);
        if (packet.eof) { //如果是结束包
            ++finished_cams; //记录：又一个摄像头读完了
            if (first_eof_frame_index < 0) {
                // 收到第一个EOF，记录帧索引（模拟单线程行为：第一个摄像头读完就停止）
                // EOF的frame_index是下一个应该读取的帧索引，实际最后一个有效帧是frame_index-1
                // 所以应该输出frame_index 0到first_eof_frame_index-1的帧批次
                first_eof_frame_index = packet.frame_index;
                stop_flag.store(true); //设置停止信号
            }
            // 收到EOF后，尝试输出已收齐的批次（可能其他摄像头已经读取了更多帧）
            try_output_complete_batches();
            continue; 
        }

        //// 如果已收到第一个EOF，且这个帧包的索引 >= EOF的帧索引，跳过（无法组成完整批次）
        //// EOF的frame_index是下一个应该读取的帧索引，实际最后一个有效帧是frame_index-1
        //// 所以frame_index >= first_eof_frame_index的帧包应该被跳过
        //if (first_eof_frame_index >= 0 && packet.frame_index >= first_eof_frame_index) {
        //    continue;
        //}
		// 记录每个摄像头的EOF帧索引
		std::map<int, int> cam_eof_indices;

		// 在EOF处理部分：
		if (packet.eof) {
			cam_eof_indices[packet.cam_id] = packet.frame_index;
			++finished_cams;

			if (first_eof_frame_index < 0) {
				first_eof_frame_index = packet.frame_index;
				stop_flag.store(true);
			}
			continue;
		}

		// 帧处理时检查：只有当该摄像头已EOF且帧索引>=其EOF索引时才跳过
		auto eof_it = cam_eof_indices.find(packet.cam_id);
		if (eof_it != cam_eof_indices.end() && packet.frame_index >= eof_it->second) {
			continue;
		}

        //正常帧包：存入缓冲区
        auto& frames_for_index = frame_buffer[packet.frame_index];
        frames_for_index.emplace(packet.cam_id, std::move(packet.frame));
        //尝试组装并输出已收齐的帧批次
        try_output_complete_batches();
    }
    
    //5、清理资源
    // 先设置停止信号，等待所有读取线程结束
    stop_flag.store(true);
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();//等待这个线程完成
        }
    }
    
    // 关闭packet_queue，这样后续的pop()调用不会一直阻塞
    packet_queue.close();
    
    // 继续处理队列中剩余的数据包，直到队列为空
    // 只处理小于first_eof_frame_index的帧包（模拟单线程行为）
    while (true) {
        auto packet_opt = packet_queue.pop();
        if (!packet_opt) {
            break; // 队列已关闭且为空
        }
        
        FramePacket packet = std::move(*packet_opt);
        if (packet.eof) {
            // 如果还有EOF包，说明之前计数有误，但这里已经不重要了
            continue;
        }
        
        // 如果已收到第一个EOF，且这个帧包的索引 >= EOF的帧索引，跳过（无法组成完整批次）
        // EOF的frame_index是下一个应该读取的帧索引，实际最后一个有效帧是frame_index-1
        // 所以frame_index >= first_eof_frame_index的帧包应该被跳过
        if (first_eof_frame_index >= 0 && packet.frame_index >= first_eof_frame_index) {
            continue;
        }
        
        //正常帧包：存入缓冲区
        auto& frames_for_index = frame_buffer[packet.frame_index];
        frames_for_index.emplace(packet.cam_id, std::move(packet.frame));
        //尝试组装并输出已收齐的帧批次
        try_output_complete_batches();
    }
    
    // 所有摄像头都读完后，再次尝试输出缓冲区中剩余的完整帧批次
    // 使用try_output_complete_batches函数来处理，它会正确处理EOF边界条件
    try_output_complete_batches();
    
    //关闭输出队列
    output_queue.close();
}


