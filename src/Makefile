include aslp.mk

SUBDIRS = warp-ctc aslp-cudamatrix aslp-segment \
          aslp-nnet aslp-nnetbin aslp-vad aslp-bin 

ifeq ($(USE_ONLINE), true) 
    SUBDIRS += aslp-online aslp-onlinebin
endif


all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
    done

mpi:
	$(MAKE) -C aslp-nnet -f Makefile.mpi; 
	$(MAKE) -C aslp-nnetbin -f Makefile.mpi; 

.PHONY: clean
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
    done

