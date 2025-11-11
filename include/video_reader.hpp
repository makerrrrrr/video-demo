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

// 视频读取任务结构体
struct VideoReadTask {
    std::string src; // 视频源路径
    std::string save_path; // 视频保存路径
    int cam_id; // 摄像头ID
    bool is_completed = false; // 是否完成
    bool is_failed = false; // 是否失败
};

// 视频读取任务管理器
class VideoTaskManager {
public:
    explicit VideoTaskManager(const std::vector<VideoReadTask>& tasks);

    std::optional<VideoReadTask> get_task();//获取任务
    void finish_task(const VideoReadTask& task);//完成任务
    void trigger_exit(); // 触发退出
    bool all_tasks_completed() const; // 是否所有任务完成
    std::map<int, VideoReadTask> get_completed_tasks() const; // 获取完成任务

private:
    mutable std::mutex mutex_; // 互斥锁
    std::condition_variable cv_; // 条件变量
    std::queue<VideoReadTask> task_queue_; // 任务队列
    std::map<int, VideoReadTask> completed_tasks_; // 完成任务
    std::atomic<bool> exit_flag_ = false; // 退出标志
    std::atomic<size_t> completed_count_ = 0; // 完成任务计数
    size_t total_tasks_ = 0; // 总任务数
};

void video_read_thread(VideoTaskManager& task_manager);

std::vector<VideoReadTask> collect_video_tasks(const std::filesystem::path& input_dir,
                                               const std::filesystem::path& output_dir);

