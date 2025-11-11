#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <queue>

#include <opencv2/core.hpp>

// 同步帧的结构体
struct FrameBatch {
    int frame_index = -1; //帧索引
    double timestamp = 0.0; //时间戳
    std::map<int, cv::Mat> frames; //帧数据

    bool is_valid() const noexcept { return !frames.empty(); } //这批帧是否有效
};

// 线程安全的阻塞队列，允许多线程环境下的安全操作，其中一个线程可以推送数据，另一个线程可以取出数据
template <typename T>
class BlockingQueue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            queue_.emplace(std::move(value));
        }
        cv_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool closed() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool closed_ = false;
};


