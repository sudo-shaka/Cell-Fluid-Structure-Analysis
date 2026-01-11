#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Simple thread pool for parallel task execution
class ThreadPool {
public:
  inline ThreadPool(); // Default constructor: uses hardware concurrency
  explicit ThreadPool(
      size_t n_threads); // Construct with specified number of threads
  inline ~ThreadPool();  // Destructor: joins all threads

  // Enqueue a task (callable object) to be executed by the pool
  template <class F> void enqueue(F &&task);

  // Wait until all tasks have finished
  inline void wait();

  size_t max_threads; // Number of worker threads

private:
  std::vector<std::thread> workers_;        // Worker threads
  std::queue<std::function<void()>> tasks_; // Task queue
  std::mutex queue_mutex_;                  // Mutex for task queue
  std::condition_variable cond_var_; // Condition variable for task notification
  std::atomic<bool> stop_ = false;   // Flag to stop the pool
  std::atomic<int> active_tasks_ = 0; // Number of currently active tasks
};

// Constructor: launches worker threads
inline ThreadPool::ThreadPool(size_t n_threads) {
  max_threads = n_threads;
  for (size_t i = 0; i < n_threads; ++i) {
    workers_.emplace_back([this]() {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          // Wait for a task or stop signal
          cond_var_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
          if (stop_ && tasks_.empty())
            return; // Exit if stopping and no tasks left
          task = std::move(tasks_.front());
          tasks_.pop();
          ++active_tasks_;
        }
        task(); // Execute the task
        std::lock_guard<std::mutex> lock(queue_mutex_);
        --active_tasks_;
        cond_var_.notify_all(); // Notify waiters in case all tasks are done
      }
    });
  }
}

// Default constructor: uses number of hardware threads
inline ThreadPool::ThreadPool()
    : ThreadPool(std::thread::hardware_concurrency()) {}

inline ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  cond_var_.notify_all();
  for (auto &worker : workers_)
    worker.join();
}

// Enqueue a new task into the pool
template <class F> void ThreadPool::enqueue(F &&task) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    tasks_.emplace(std::forward<F>(task));
  }
  cond_var_.notify_one();
}

// Wait until all tasks are finished
inline void ThreadPool::wait() {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  cond_var_.wait(lock,
                 [this]() { return tasks_.empty() && active_tasks_ == 0; });
}

// Utility function: parallel for loop using the thread pool
template <typename F>
void parallel_for(ThreadPool &pool, size_t count, F &&func) {
  size_t n_chunks = pool.max_threads * 4;
  size_t chunk_size = (count + n_chunks - 1) / n_chunks;
  for (size_t c = 0; c < n_chunks; ++c) {
    size_t start = c * chunk_size;
    size_t end = std::min(start + chunk_size, count);
    if (start < end) {
      pool.enqueue([=, &func]() {
        for (size_t i = start; i < end; ++i) {
          func(i);
        }
      });
    }
  }
  pool.wait();
}
