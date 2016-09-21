include aslp.mk

SUBDIRS = aslp-cudamatrix aslp-segment \
          aslp-nnet aslp-nnetbin aslp-vad aslp-bin 

# Warp-CTC
ifeq ($(USE_WARP_CTC), true)
    SUBDIRS += warp-ctc
endif

# MPI parallel
ifeq ($(USE_MPI), true)
    SUBDIRS += aslp-parallel aslp-parallelbin
endif

# Online
ifeq ($(USE_ONLINE), true) 
    SUBDIRS += aslp-online aslp-onlinebin
endif

all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
    done

.PHONY: clean

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
    done

