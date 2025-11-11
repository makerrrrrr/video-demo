#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

struct VideoReadTask {
    std::string src;
    std::string save_path;
    int cam_id;
    bool is_completed = false;
    bool is_failed = false;
};

class VideoTaskManager {
public:
    explicit VideoTaskManager(const std::vector<VideoReadTask>& tasks);

    std::optional<VideoReadTask> get_task();
    void finish_task(const VideoReadTask& task);
    void trigger_exit();
    bool all_tasks_completed() const;
    std::map<int, VideoReadTask> get_completed_tasks() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<VideoReadTask> task_queue_;
    std::map<int, VideoReadTask> completed_tasks_;
    std::atomic<bool> exit_flag_ = false;
    std::atomic<size_t> completed_count_ = 0;
    size_t total_tasks_ = 0;
};

void video_read_thread(VideoTaskManager& task_manager);

std::vector<VideoReadTask> collect_video_tasks(const std::filesystem::path& input_dir,
                                               const std::filesystem::path& output_dir);

