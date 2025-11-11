#include "video_reader.hpp"

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

VideoTaskManager::VideoTaskManager(const std::vector<VideoReadTask>& tasks) {
    for (const auto& task : tasks) {
        task_queue_.push(task);
    }
    total_tasks_ = tasks.size();
}

std::optional<VideoReadTask> VideoTaskManager::get_task() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !task_queue_.empty() || exit_flag_; });
    if (exit_flag_ && task_queue_.empty()) {
        return std::nullopt;
    }
    auto task = task_queue_.front();
    task_queue_.pop();
    return task;
}

void VideoTaskManager::finish_task(const VideoReadTask& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_tasks_[task.cam_id] = task;
    completed_count_++;
}

void VideoTaskManager::trigger_exit() {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_flag_ = true;
    cv_.notify_all();
}

bool VideoTaskManager::all_tasks_completed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_count_ == total_tasks_;
}

std::map<int, VideoReadTask> VideoTaskManager::get_completed_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_tasks_;
}

void video_read_thread(VideoTaskManager& task_manager) {
    while (true) {
        auto opt_task = task_manager.get_task();
        if (!opt_task) {
            break;
        }
        auto task = *opt_task;

        cv::VideoCapture cap(task.src);
        if (!cap.isOpened()) {
            task.is_failed = true;
            task_manager.finish_task(task);
            continue;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const cv::Size frame_size(width, height);
        const int preferred_fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        const int fallback_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

        cv::VideoWriter writer(task.save_path, preferred_fourcc, fps, frame_size, true);
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
        while (cap.read(frame)) {
            writer.write(frame);
        }

        cap.release();
        writer.release();
        task.is_completed = true;
        task_manager.finish_task(task);
    }
}

std::vector<VideoReadTask> collect_video_tasks(const std::filesystem::path& input_dir,
                                               const std::filesystem::path& output_dir) {
    std::vector<VideoReadTask> tasks;
    int cam_id = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto& src_path = entry.path();
        const auto ext = src_path.extension().string();
        if (ext != ".mp4" && ext != ".MP4" && ext != ".avi" && ext != ".AVI" && ext != ".mov" && ext != ".MOV") {
            continue;
        }

        auto relative_path = std::filesystem::relative(src_path, input_dir);
        auto dest_path = output_dir / relative_path;
        std::filesystem::create_directories(dest_path.parent_path());

        tasks.push_back({src_path.string(), dest_path.string(), cam_id++});
    }
    return tasks;
}

