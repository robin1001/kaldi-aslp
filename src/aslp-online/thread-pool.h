// aslp-online/thread-pool.h

/* Created on 2016-05-24
 * Author: Binbin Zhang
 */

#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <unistd.h>
#include <pthread.h>

#include <vector>
#include <queue>
#include <map>

static void ErrorExit(const char *msg) {
    perror(msg);
    exit(1);
}

// Thread interface for thread class
class Threadable {
public:
    // @resource: thread resource allocated by the thread pool
    virtual void operator() (void *resource) = 0;
    virtual ~Threadable() {}
};

// ThreadPool
class ThreadPool {
public:
    //@param[in] resource_pool, optional resource pool for every thread
    //           here we just use the void * for polymorphism, for it is simple and stupid
    //           we can also use the template programming, like template <typename C> class ThreadPool
    //           but it is more complicated and requires specific init
    ThreadPool(int num_thread = 5, std::vector<void *> *resource_pool = NULL): 
            num_thread_(num_thread), 
            stop_(false) {
        if (resource_pool != NULL && resource_pool->size() != num_thread_) {
            ErrorExit("resource and num thread must equal");
        }
        if (pthread_mutex_init(&mutex_, NULL) != 0) {
            ErrorExit("mutex init error");
        }
        if (pthread_cond_init(&cond_, NULL) != 0) {
            ErrorExit("cond init error");
        }
        // Create num_thread thread at once
        threads_.resize(num_thread_);
        for (int i = 0; i < threads_.size(); i++) {
            if (pthread_create(&threads_[i], NULL, 
                               ThreadPool::WorkerThread, (void *)this) != 0) {
                ErrorExit("pthread create error");
            }
            if (resource_pool != NULL) {
                resource_table_[threads_[i]] = (*resource_pool)[i];
            } else {
                resource_table_[threads_[i]] = NULL;
            }
        }
    }

    ~ThreadPool() {
        pthread_mutex_lock(&mutex_);
        stop_ = true;
        pthread_mutex_unlock(&mutex_);
        // notify all thread to stop
        pthread_cond_broadcast(&cond_);

        for (int i = 0; i < threads_.size(); i++) {
            pthread_join(threads_[i], NULL);
        }

        pthread_mutex_destroy(&mutex_);
        pthread_cond_destroy(&cond_);
    }

    void *GetResource(pthread_t tid) {
        assert(resource_table_.find(tid) != resource_table_.end());
        return resource_table_[tid];
    }

    void AddTask(Threadable *task) {
        pthread_mutex_lock(&mutex_);
        task_queue_.push(task);
        pthread_mutex_unlock(&mutex_);
        pthread_cond_signal(&cond_);
    }

    // Wait a task to execute  
    Threadable *WaitTask() {
        Threadable *task = NULL;
        pthread_mutex_lock(&mutex_);
        while (!stop_ && task_queue_.empty()) {
            pthread_cond_wait(&cond_, &mutex_);
        }
        if (task_queue_.size() > 0) {
            task = task_queue_.front();
            task_queue_.pop();
        }
        // else stop_ = true return NULL
        pthread_mutex_unlock(&mutex_);
        return task;
    }

    // PoolWorker thread
    static void *WorkerThread(void *arg) {
        sleep(3); // wait main thread to construct the resource_table_
        ThreadPool *pool = static_cast<ThreadPool *>(arg);
        void *resource = pool->GetResource(pthread_self());
        for(;;) {
            Threadable *task = pool->WaitTask();
            // Stop
            if (task == NULL) break;
            else {
                (*task)(resource); // Run the task
                delete task;
            }
        }
        return NULL;
    }

private:
    int num_thread_;
    bool stop_;
    std::vector<pthread_t> threads_;
    std::vector<void *> *resource_pool_;
    std::queue<Threadable *> task_queue_; //TaskQueue
    std::map<pthread_t, void *> resource_table_;
    pthread_cond_t cond_;
    pthread_mutex_t mutex_;
};

#endif
