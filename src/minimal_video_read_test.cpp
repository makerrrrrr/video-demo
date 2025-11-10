#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

// 视频读取任务结构体（存储单路视频的源信息和存储路径）
struct VideoReadTask {
    std::string src;         // 视频源（本地路径：./cam0.mp4；RTSP流：rtsp://xxx）
    std::string save_path;   // 本地存储路径（如./saved_videos/cam0.mp4）
    int cam_id;              // 相机ID（用于区分多路视频）
    bool is_completed = false; // 任务是否完成
    bool is_failed = false;     // 任务是否失败
};

// 全局任务管理器（线程安全，用于分发任务和监控状态）
// 多路视频（本地/RTSP）读取任务管理器,每个任务独立处理一条流
class VideoTaskManager {
public:
// 加上explicit为了避免隐式转换， VideoTaskManager(tasks) 才能调用
    explicit VideoTaskManager(const std::vector<VideoReadTask>& tasks) {
        for (const auto& task : tasks) {
            task_queue_.push(task);
        }
        total_tasks_ = tasks.size();
    }

    // 线程从队列取任务（阻塞直到有任务或退出）
    // std::optional<VideoReadTask> 表示可能返回一个VideoReadTask，也可能返回一个空值
    std::optional<VideoReadTask> get_task() {
        std::unique_lock<std::mutex> lock(mutex_); // 加锁，防止多个线程同时访问task_queue_
        // 在条件变量上等待，直到有任务或退出
        cv_.wait(lock, [this]() { return !task_queue_.empty() || exit_flag_; });
        if (exit_flag_ && task_queue_.empty()) {
            return std::nullopt; // 如果退出标志为true且任务队列为空，返回空值
        }
        auto task = task_queue_.front(); // 获取任务队列的第一个任务
        task_queue_.pop(); // 移除任务队列的第一个任务
        return task; // 返回任务
    }

    // 任务完成后更新状态
    void finish_task(const VideoReadTask& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        completed_tasks_[task.cam_id] = task;
        completed_count_++;
    }

    // 触发退出（停止所有读取线程）
    void trigger_exit() {
        std::lock_guard<std::mutex> lock(mutex_);
        exit_flag_ = true;
        cv_.notify_all();
    }

    // 检查所有任务是否完成
    bool all_tasks_completed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_count_ == total_tasks_;
    }

    // 获取所有完成的任务（供后续使用）
    std::map<int, VideoReadTask> get_completed_tasks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_tasks_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<VideoReadTask> task_queue_;
    std::map<int, VideoReadTask> completed_tasks_;
    std::atomic<bool> exit_flag_ = false;
    std::atomic<size_t> completed_count_ = 0;
    size_t total_tasks_ = 0;
};

// 视频读取线程函数（每个线程处理一个任务）
// 每个工作线程要执行的函数
void video_read_thread(VideoTaskManager& task_manager) {
    while (true) {
        auto opt_task = task_manager.get_task();
        if (!opt_task) {
            break;
        }
        auto task = *opt_task;

        // cap是VideoCapture类的一个实例，用于读取视频
        // task.src是视频源，可以是本地路径或RTSP流
        cv::VideoCapture cap(task.src);
        if (!cap.isOpened()) {
            task.is_failed = true; // 设置任务失败
            task_manager.finish_task(task); // 完成任务
            continue; // 继续下一个任务
        }

        // 获取视频的帧率、宽度和高度
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        // 将前面获取的width和height封装成cv::Size类型
        const cv::Size frame_size(width, height);
        //  cv::VideoWriter::fourcc 是OpenCV中用于指定视频编码的函数
        // H.264编码，最常用的视频编码格式
        const int preferred_fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        //如果打开失败就回退到MJPG编码
        const int fallback_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        // cv::VideoWriter writer 尝试用H.264编码把帧写到task.save_path
        cv::VideoWriter writer(task.save_path, preferred_fourcc, fps, frame_size, true);
        // 如果打开失败就回退到MJPG编码
        if (!writer.isOpened()) {
            writer.open(task.save_path, fallback_fourcc, fps, frame_size, true);
            if (!writer.isOpened()) {
                cap.release();
                task.is_failed = true;
                task_manager.finish_task(task);
                continue;
            }
        }

        cv::Mat frame;
        //  while (cap.read(frame)) 循环从VideoCapture中读取一帧，并赋值给frame
        while (cap.read(frame)) {
            writer.write(frame); // 将frame写入到writer中,false表示读取不到新帧（视频结束或出错），循环就会终止
        }

        cap.release(); // 释放VideoCapture
        writer.release(); // 释放VideoWriter
        task.is_completed = true; // 设置任务完成
        task_manager.finish_task(task); // 完成任务
    }
}

namespace {

bool create_test_video(const std::filesystem::path& path, int cam_id) {
    std::filesystem::create_directories(path.parent_path());
    cv::VideoWriter writer(path.string(), cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 15.0, cv::Size(320, 240), true);
    if (!writer.isOpened()) {
        return false;
    }

    for (int i = 0; i < 30; ++i) {
        cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(10 * cam_id, (i * 5) % 255, (i * 3) % 255));
        writer.write(frame);
    }

    writer.release();
    return true;
}

} // namespace

int main() {
    const std::filesystem::path input_dir = "test_inputs";
    const std::filesystem::path output_dir = "saved_videos";
    std::filesystem::create_directories(input_dir);
    std::filesystem::create_directories(output_dir);

    const auto src0 = input_dir / "cam0.mp4";
    const auto src1 = input_dir / "cam1.mp4";

    if (!create_test_video(src0, 0) || !create_test_video(src1, 1)) {
        std::cerr << "生成测试视频失败。" << std::endl;
        return 1;
    }

    std::vector<VideoReadTask> tasks = {
        {src0.string(), (output_dir / "cam0.mp4").string(), 0},
        {src1.string(), (output_dir / "cam1.mp4").string(), 1},
    };

    VideoTaskManager task_manager(tasks);

    const size_t thread_count = 2;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        workers.emplace_back(video_read_thread, std::ref(task_manager));
    }

    while (!task_manager.all_tasks_completed()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    task_manager.trigger_exit();

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    auto completed = task_manager.get_completed_tasks();
    bool all_ok = true;
    for (const auto& [cam_id, task] : completed) {
        if (task.is_failed || !task.is_completed || !std::filesystem::exists(task.save_path)) {
            std::cerr << "Cam" << cam_id << " 处理失败。" << std::endl;
            all_ok = false;
        }
    }

    if (!all_ok || completed.size() != tasks.size()) {
        std::cerr << "任务未全部成功完成。" << std::endl;
        return 1;
    }

    std::cout << "多线程视频读取测试通过。" << std::endl;
    return 0;
}

