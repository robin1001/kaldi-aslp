/* Created on 2016-07-26
 * Author: Zhang Binbin
 */

#ifndef ASLP_PARALLEL_ITF_H_
#define ASLP_PARALLEL_ITF_H_

#include "aslp-parallel/mpi-node.h"

namespace kaldi {

// Mpi tag type 
typedef enum {
    kTagMsg = 0,
    kTagModel = 1
} MpiTagType;

// Mpi message types in kTagMsg for server and worker communication
typedef enum {
    kMsgSynchronize = 0x00,
    kMsgFinished = 0x01
} MpiMsgType;


class IWorker : public MpiNode {
public:
    virtual ~IWorker() {}
    virtual void InitParam(const std::vector<std::pair<BaseFloat *, int> > &params) = 0; 
    // @params: num_worker_samples, new sample frames since last synchronization
    // return true if all worker finished their own data
    virtual bool Synchronize(int num_worker_samples) = 0;
    // Wait other workers
    virtual void Stop() = 0;
};

class IServer : public MpiNode {
public:
    virtual ~IServer() {}
    virtual void InitParam(const std::vector<std::pair<BaseFloat *, int> > &params) = 0; 
    virtual void Run() = 0;
};

} // namespace kaldi

#endif
