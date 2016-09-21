/* Created on 2016-07-26
 * Author: Zhang Binbin
 */


#ifndef ASLP_PARALLEL_MPI_NODE_H_
#define ASLP_PARALLEL_MPI_NODE_H_

#include "mpi.h"

namespace kaldi {

// mpi wrapper
class MpiNode {
public:
    MpiNode() {
        int argc;
        char **argv = NULL;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes_);
    }

    virtual ~MpiNode() {
        MPI_Finalize();
    }

    int Rank() const {
        return rank_;
    }

    int NumNodes() const {
        return num_nodes_;
    }

    int MainNode() const {
        return 0;
    }

    bool IsMainNode() const {
        return rank_ == 0;
    }

    static MPI_Datatype GetDataType(char *) {   
        return MPI_CHAR;
    }   
    static MPI_Datatype GetDataType(int *) {   
        return MPI_INT;
    }   
    static MPI_Datatype GetDataType(float *) {   
        return MPI_FLOAT;
    }   
    static MPI_Datatype GetDataType(double *) {   
        return MPI_DOUBLE;
    }   
    static MPI_Datatype GetDataType(size_t *) {   
        return sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_LONG_LONG_INT;
    }
	
	void Barrier() const {
        MPI_Barrier(MPI_COMM_WORLD);
	}
	
	template <class ElemType> 
	void AllReduce(ElemType *data, int size) {
		if (num_nodes_ > 0) {
            MPI_Allreduce(MPI_IN_PLACE, data, size, GetDataType(data), 
                MPI_SUM, MPI_COMM_WORLD);
        }
	}

protected:
    int rank_, num_nodes_;
};

} // namespace kaldi

#endif
