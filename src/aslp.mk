# Optional make warp-ctc
USE_CTC = true
USE_WARP_CTC = false

# Optinal make mpi parallel(aslp-paralle aslp-parallelbin)
USE_MPI = true

# Optinal make online part, Online depends on crf++, add it's lib and include
USE_ONLINE = false

CRF_ROOT = /home/zhangbinbin/zhangbinbin/kaldi-aslp/src/crf
CRF_FLAGS = /home/zhangbinbin/zhangbinbin/kaldi-aslp/src/crf/libcrfpp.a
