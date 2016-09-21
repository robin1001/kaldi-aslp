CXX = g++
OBJS = forward-max-match.o
BINS = aslp-forward-max-match-segment

all: $(BINS) $(OBJS)


aslp-forward-max-match-segment: aslp-forward-max-match-segment.cc forward-max-match.o
	$(CXX) $^ -o $@

forward-max-match.o: forward-max-match.h


.PHONY: clean

clean:
	rm -f *.o $(BINS)
