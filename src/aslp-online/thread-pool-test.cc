/* Created on 2016-05-24
 * Author: Binbin Zhang
 */

#include <stdio.h>
#include <stdlib.h>

#include "thread-pool.h"

class MyTask : public Threadable {
public:
    MyTask(int id): id_(id) {}
    ~MyTask() {
        //printf("thread %d exit\n", id_);
    }
    virtual void operator() (void *resource) {
        int num = rand() % 100000;
        printf("thread %d num %d\n", id_, num);
        for (int i = 0; i < num; i++);
    }
private:
    int id_;
};

void TestThreadPool() {
    ThreadPool thread_pool(5);
    for (int i = 0; i < 100; i++) {
       Threadable *task = new MyTask(i);
       thread_pool.AddTask(task);
    }
}


int main() {
    TestThreadPool();
    return 0;
}



