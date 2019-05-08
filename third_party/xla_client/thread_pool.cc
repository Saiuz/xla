#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>

#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace xla {
namespace env {
namespace {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads) {
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this]() { Worker(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = true;
      cv_.notify_all();
    }
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void Schedule(std::function<void()> closure) {
    if (threads_.empty()) {
      std::thread thread(std::move(closure));
      thread.detach();
    } else {
      std::lock_guard<std::mutex> lock(mutex_);
      work_.emplace_back(std::move(closure));
      cv_.notify_one();
    }
  }

 private:
  void Worker() {
    while (true) {
      std::function<void()> closure = GetWork();
      if (closure == nullptr) {
        break;
      }
      try {
        closure();
      } catch (const std::exception& ex) {
        XLA_COUNTER("ThreadPoolException", 1);
        TF_LOG(ERROR) << "Exception from running thread pool closure: "
                      << ex.what();
      }
    }
  }

  std::function<void()> GetWork() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return done_ || !work_.empty(); });
    if (work_.empty()) {
      return nullptr;
    }
    std::function<void()> closure(std::move(work_.front()));
    work_.pop_front();
    return closure;
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool done_ = false;
  std::deque<std::function<void()>> work_;
};

ThreadPool* GetThreadPool() {
  static size_t num_threads = sys_util::GetEnvInt(
      "XLA_THREAD_POOL_SIZE", std::thread::hardware_concurrency());
  static ThreadPool* pool = new ThreadPool(num_threads);
  return pool;
}

ThreadPool* GetIoThreadPool() {
  // For the I/O thread pool, create one which schedules by creating new
  // threads. Since I/O operations are usually long lasting, wasting 10s of
  // microseconds in creating new threads does not hurt, and as a plus, we will
  // never get into the typical thread-pool-size-deadlock.
  static ThreadPool* pool = new ThreadPool(/*num_threads=*/0);
  return pool;
}

}  // namespace

class Completion::Data {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return completed_; });
    if (exptr_ != nullptr) {
      std::rethrow_exception(exptr_);
    }
  }

  static std::function<void()> GetCompleter(std::shared_ptr<Data> data,
                                            std::function<void()> closure) {
    auto closure_wrapper = [closure = std::move(closure), data]() {
      std::exception_ptr exptr;
      try {
        closure();
      } catch (...) {
        exptr = std::current_exception();
      }
      data->Complete(exptr);
    };
    return closure_wrapper;
  }

 private:
  void Complete(std::exception_ptr exptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    exptr_ = std::move(exptr);
    completed_ = true;
    cv_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exptr_;
};

Completion::Completion(std::shared_ptr<Data> data) : data_(std::move(data)) {}

Completion::~Completion() {}

void Completion::Wait() { data_->Wait(); }

void ScheduleClosure(std::function<void()> closure) {
  GetThreadPool()->Schedule(std::move(closure));
}

void ScheduleIoClosure(std::function<void()> closure) {
  GetIoThreadPool()->Schedule(std::move(closure));
}

Completion ScheduleClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  GetThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  return Completion(std::move(data));
}

Completion ScheduleIoClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  GetIoThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  return Completion(std::move(data));
}

}  // namespace env
}  // namespace xla
