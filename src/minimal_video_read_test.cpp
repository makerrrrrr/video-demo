#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include "video_reader.hpp"

int main() {
    const std::filesystem::path input_dir = std::filesystem::path("D:/code/VGGTSyncMultiCam-Demo/video_test");
    const std::filesystem::path output_dir = "saved_videos";
    std::filesystem::create_directories(output_dir);

    if (!std::filesystem::exists(input_dir)) {
        std::cerr << "输入目录不存在: " << input_dir << std::endl;
        return 1;
    }

    auto tasks = collect_video_tasks(input_dir, output_dir);

    if (tasks.empty()) {
        std::cerr << "在目录 " << input_dir << " 中未找到可用视频文件。" << std::endl;
        return 1;
    }

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

