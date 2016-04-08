
SUBDIRS = aslp-bin aslp-cudamatrix aslp-nnet aslp-nnetbin

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

