UNAME_S := $(shell uname -s)

TARGET = mpitest
LIBS = -lm
CC = mpic++
CXXFLAGS = -g -O3 -Wall -std=c++11
CXXFLAGS += -I/usr/mpi/gcc/mvapich2-2.2a/include

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
