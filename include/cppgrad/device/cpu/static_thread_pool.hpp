#ifndef CPPGRAD_STATIC_THREAD_POOL_HPP
#define CPPGRAD_STATIC_THREAD_POOL_HPP

#include <functional>
#include <queue>
#include <thread>
#include <vector>

#include <condition_variable>

#include <mutex>

namespace cppgrad::util {

class StaticThreadPool {
public:
    ~StaticThreadPool()
    {
        terminatePool = true; // use this flag in condition.wait
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            while (!taskQueue.empty())
                taskQueue.pop();
        }

        cvTask.notify_all(); // wake up all threads.
        // Join all threads.
        for (std::thread& th : threads) {
            th.join();
        }

        threads.clear();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        cvFinished.wait(lock, [this]() { return taskQueue.empty() && (tasksWorking == 0); });
    }

    using Task = std::function<void()>;

    template <typename TaskType>
    void add_task(TaskType&& newTask)
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        taskQueue.push(std::forward<TaskType>(newTask));
        cvTask.notify_one();
    }

    static StaticThreadPool& get()
    {
        thread_local static StaticThreadPool pool;
        return pool;
    }

private:
    StaticThreadPool()
    {
        for (int i = 0; i < std::thread::hardware_concurrency(); i++) {
            threads.emplace_back(&StaticThreadPool::infiniteFunction, this); // PVS V823; push_back -> emplace_back
        }
    }

    StaticThreadPool(const StaticThreadPool&) = delete;
    StaticThreadPool(StaticThreadPool&&) = delete;

    void infiniteFunction()
    {
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);

            cvTask.wait(lock, [this]() {
                return !taskQueue.empty() || terminatePool;
            });

            if (terminatePool) {
                tasksWorking = 0;
                cvFinished.notify_all();

                lock.unlock(); // PVS V1020
                return;
            }

            if (!taskQueue.empty()) {
                tasksWorking++;

                Task job = taskQueue.front();
                taskQueue.pop();

                lock.unlock();

                job();

                lock.lock();

                tasksWorking--;

                cvFinished.notify_one();
            }
        }
    }

    bool terminatePool = false;

    int tasksWorking = 0;

    std::mutex queueMutex;

    std::condition_variable cvTask;
    std::condition_variable cvFinished;

    std::vector<std::thread> threads;
    std::queue<Task> taskQueue;
};

}

#endif