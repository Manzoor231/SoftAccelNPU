#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace softaccelnpu {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Submit a task to the pool
    void enqueue(std::function<void()> task);

    // Parallel loop utility
    // Splits range [start, end) into chunks and executes in parallel
    void parallel_for(size_t start, size_t end, std::function<void(size_t, size_t)> chunk_func);

    size_t num_threads() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

// Global accessor for the runtime thread pool
ThreadPool& get_thread_pool();

} // namespace softaccelnpu
