/* Created on 2016-07-26
 * Author: Zhang Binbin
 */


#ifndef ASLP_PARALLEL_MPI_NODE_H_
#define ASLP_PARALLEL_MPI_NODE_H_

#include "mpi.h"
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

namespace kaldi {

// mpi wrapper
class MpiNode {
public:
    MpiNode() {
        int argc = 0;
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

    /// Acc stats for global batch normalization 
    void ReduceAccStat(const std::vector<double *> &acc_params, 
                       const std::vector<std::pair<double*, int> > &data_params) {
        Barrier();
        for (int i = 0; i < acc_params.size(); i++) {
            AllReduce(acc_params[i], 1); 
        }

        for (int i = 0; i < data_params.size(); i++) {
            CuSubVector<double> gpu_data(data_params[i].first, 
                                            data_params[i].second);
            Vector<double> cpu_data(data_params[i].second);
            cpu_data.CopyFromVec(gpu_data);
            AllReduce(cpu_data.Data(), cpu_data.Dim());
            gpu_data.CopyFromVec(cpu_data);
        }
    }


protected:
    int rank_, num_nodes_;
};

} // namespace kaldi

#endif
