
all: cl_maxalloc

cl_maxalloc: cl_maxalloc.c
	gcc -lOpenCL -o $@ $<

clean:
	rm -vf cl_maxalloc
