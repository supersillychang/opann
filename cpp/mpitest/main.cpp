//
//  main.cpp
//  mpitest
//
//  Created by Chang Sun on 2/28/18.
//  Copyright © 2018 Chang Sun. All rights reserved.
//

#include <iostream>
#include <string>
#include <mpi.h>

#include "nanotimer.h"

#define MPI_MASTER  0

typedef float   data_t;

int main (int argc, const char *argv[])
{
    MPI_Init(NULL, NULL);
    
    int worker_count;
    int worker_rank;
    
    MPI_Comm_size(MPI_COMM_WORLD, &worker_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_rank);
     
    int size = atoi(argv[1]);
    int iters = 1048576 / size;     // 2^20
    if (iters < 10) {
        iters = 10;
    }
    
    int trials = 10;
    int warmup = 1;
    
    if (worker_rank == MPI_MASTER) {
        std::cout << "Process count: " << worker_count << std::endl;
        std::cout << "Problem size: " << size << std::endl;
        std::cout << "Iterations per trial: " << iters << std::endl;
        std::cout << std::endl;
    }
    
    
    data_t* bcast = new data_t[size];
    if (worker_rank == MPI_MASTER) {
        for (int i = 0; i < size; i++) {
            bcast[i] = 0.1f;
        }
    }
    
    data_t* sum_local = new data_t[size];
    data_t* sum_global = NULL;
    for (int i = 0; i < size; i++) {
        sum_local[i] = 0.1f;
    }
    if (worker_rank == MPI_MASTER) {
        sum_global = new data_t[size];
    }
    
    NanoTimer timer_bcast;
    NanoTimer timer_reduce;
    
    uint64_t sec, nano;
    
    for (int t = 0; t < trials + warmup; t++) {
        
        // MPI_Bcast
        timer_bcast.restart(); timer_bcast.stop();
        for (int i = 0; i < iters; i++) {
            if (worker_rank == MPI_MASTER) {
                for (int j = 0; j < size; j++) {
                    bcast[j] = 1.0 * i / iters + 1.0 * j / size;
                }
            }
            timer_bcast.resume();
            MPI_Bcast(bcast, size, MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
            timer_bcast.stop();
        }
        //timer_bcast.stop();
        
        // MPI_Reduce
        timer_reduce.restart(); timer_reduce.stop();
        for (int i = 0; i < iters; i++) {
            if (worker_rank == MPI_MASTER) {
                for (int j = 0; j < size; j++) {
                    sum_local[j] = 1.0 * i / iters + 1.0 * j / size;
                }
            }
            timer_reduce.resume();
            MPI_Reduce(sum_local, sum_global, size, MPI_FLOAT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);
            timer_reduce.stop();
        }
        //timer_reduce.stop();
        
        if (worker_rank == MPI_MASTER && t >= warmup) {
            std::cout << "Trial " << t - warmup << ":" << std::endl;
            timer_bcast.getElapsed(sec, nano);
            std::cout << "MPI_Bcast time (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) / iters << std::endl;
            timer_reduce.getElapsed(sec, nano);
            std::cout << "MPI_Reduce time (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) / iters << std::endl;
            std::cout << std::endl;
        }
    }
    
    delete [] bcast;
    delete [] sum_local;
    if (sum_global != NULL) {
        delete [] sum_global;
    }
    
    MPI_Finalize();
}
