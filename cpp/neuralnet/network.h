//
//  network.h
//  neuralnet
//
//  Created by Chang Sun on 1/2/18.
//  Copyright © 2018 Chang Sun. All rights reserved.
//

#ifndef network_h
#define network_h

#include <iostream>
#include <mpi.h>
#include "oclworkengine.h"
#include "nanotimer.h"

// Define OclWorkEngine parameters
#define OCL_PROGRAM             "./kernels.cl"
#define SIGMOID_KERNEL          "sigmoid"
#define SIGMOID_PRIME_KERNEL    "sigmoid_prime"
#define VEC_MULT_KERNEL         "vec_mult"
#define MAT_VEC_ADD_KERNEL      "mat_vec_add"
#define MAT_TRANS_KERNEL        "mat_trans"
#define MAT_MULT_KERNEL         "mat_mult"

// Define MPI parameters
#define MPI_MASTER              0

typedef float       data_t;

class Network
{
public:
    Network(const int* sizes, int num_layers, bool useOCL = false, bool useMPI = false, std::ostream* out = &std::cout);
    ~Network();
    void load_dataset(const char* train_file_name, const char* test_file_name, int train_data_size, int test_data_size);
    void SGD(int epochs, int mini_batch_size, data_t eta, data_t lmbda);
    
private:
    void clean_network();
    void load_train_file(const char* file_name);
    void load_test_file(const char* file_name);
    void bcast_network();
    void feedforward();
    void update_mini_batch(data_t eta, data_t lmbda);
    void cost_derivative(const data_t* y);
    void backprop(const data_t* y);
    void reduce_gradients();
    int evaluate(const data_t* test_data_x, const data_t* test_data_y);
    
    bool useOCL;
    OclWorkEngine* engine = NULL;
    
    bool useMPI;
    int worker_count = 1;
    int worker_rank = MPI_MASTER;
    
    std::ostream* debug_out;
    
    int num_layers;
    int* sizes = NULL;
    int mini_batch_size;
    int worker_batch_size;
    data_t** biases = NULL;
    data_t** weights = NULL;
    
    data_t** activations = NULL;
    data_t** primes = NULL;
    data_t** nabla_b = NULL;
    data_t** nabla_w = NULL;
    data_t** nabla_b_sum = NULL;
    data_t** nabla_w_sum = NULL;
    
    int train_data_size;
    int test_data_size;
    int worker_train_size;
    int worker_test_size;
    data_t* train_data_x = NULL;
    data_t* train_data_y = NULL;
    data_t* test_data_x = NULL;
    data_t* test_data_y = NULL;
};

#endif /* network_h */
