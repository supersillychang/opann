UNAME_S := $(shell uname -s)

TARGET = neuralnet
LIBS = -lm
ifeq ($(UNAME_S), Linux)
LIBS += -lOpenCL
endif
ifeq ($(UNAME_S), Darwin)
LIBS += -framework OpenCL
endif
CC = mpic++
CXXFLAGS = -g -O3 -Wall -std=c++11
CXXFLAGS += -I/groups/arc/apps/archive/cascades/opt/apps/cuda/8.0.61/include #OpenCL
CXXFLAGS += -I/usr/mpi/gcc/mvapich2-2.2a/include    #MPI

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
