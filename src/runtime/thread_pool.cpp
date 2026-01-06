#include "softaccelnpu/thread_pool.h"
#include <iostream>
#include <vector>
#include <future>

namespace softaccelnpu {

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    this->condition_.wait(lock, [this] {
                        return this->stop_ || !this->tasks_.empty();
                    });

                    if (this->stop_ && this->tasks_.empty())
                        return;

                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable())
            worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks_.emplace(std::move(task));
    }
    condition_.notify_one();
}

void ThreadPool::parallel_for(size_t start, size_t end, std::function<void(size_t, size_t)> chunk_func) {
    if (start >= end) return;

    size_t total_work = end - start;
    size_t num_workers = workers_.size();
    
    // Simple heuristic: if work is small, don't spawn tasks
    // But for GEMM we assume large work.
    
    // Divide work
    size_t work_per_thread = (total_work + num_workers - 1) / num_workers;
    
    std::vector<std::future<void>> futures;
    
    for (size_t i = 0; i < num_workers; ++i) {
        size_t chunk_start = start + i * work_per_thread;
        size_t chunk_end = std::min(end, chunk_start + work_per_thread);

        if (chunk_start >= end) break;

        // We use std::packaged_task to wait for completion.
        // Actually simpler: just use enqueue and a latch/counter mechanism?
        // Or std::future.
        
        auto task = std::make_shared<std::packaged_task<void()>>(
            [chunk_start, chunk_end, &chunk_func]() {
                chunk_func(chunk_start, chunk_end);
            }
        );
        
        futures.emplace_back(task->get_future());
        
        enqueue([task]() {
            (*task)();
        });
    }

    // Wait for all to complete
    for (auto& f : futures) {
        f.get();
    }
}

// Global instance
ThreadPool& get_thread_pool() {
    static ThreadPool pool(0); // Auto-detect
    return pool;
}

} // namespace softaccelnpu
