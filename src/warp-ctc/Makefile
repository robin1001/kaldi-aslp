# Created on 2016-04-11
# Author: Binbin Zhang

all: warpctc.a

warpctc.a:
	mkdir -p build;
	cd build; cmake ..; make;
	cp build/libwarpctc.a warp-ctc.a

.PHONY: clean

clean:
	rm -f warpctc.a
