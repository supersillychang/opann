//
//  network.cpp
//  neuralnet
//
//  Created by Chang Sun on 1/2/18.
//  Copyright © 2018 Chang Sun. All rights reserved.
//

#include <fstream>
#include <random>
#include <algorithm>
#include <math.h>
#include <string>
#include <map>

#include "network.h"
#include "nanotimer.h"


void sigmoid(data_t* ret_val, int size, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        for (int i = 0; i < size; i++) {
            ret_val[i] = 1.0 / (1.0 + exp(-ret_val[i]));
        }
    } else {
        size_t globalWorkSize[1] = {(size_t)size};
        engine->setKernelArg(SIGMOID_KERNEL, 0, CL_MEM_READ_WRITE, size * sizeof(data_t), ret_val, CL_TRUE);
        engine->enqueueKernel(SIGMOID_KERNEL, 1, globalWorkSize, NULL, true);
        engine->enqueueReadBuffer(SIGMOID_KERNEL, 0, size * sizeof(data_t), ret_val);
    }
}

void sigmoid_prime(data_t* ret_val, const data_t* z, int size, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        data_t sigmoid;
        for (int i = 0; i < size; i++) {
            sigmoid = 1.0 / (1.0 + exp(-z[i]));
            ret_val[i] = sigmoid * (1.0 - sigmoid);
        }
    } else {
        size_t globalWorkSize[1] = {(size_t)size};
        engine->setKernelArg(SIGMOID_PRIME_KERNEL, 0, CL_MEM_READ_WRITE, size * sizeof(data_t), ret_val, CL_TRUE);
        engine->setKernelArg(SIGMOID_PRIME_KERNEL, 1, CL_MEM_READ_ONLY, size * sizeof(data_t), z, CL_TRUE);
        engine->enqueueKernel(SIGMOID_PRIME_KERNEL, 1, globalWorkSize, NULL, true);
        engine->enqueueReadBuffer(SIGMOID_PRIME_KERNEL, 0, size * sizeof(data_t), ret_val);
    }
}

void vec_mult(data_t* ret_vec, const data_t* vec, int size, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        for (int i = 0; i < size; i++) {
            ret_vec[i] *= vec[i];
        }
    } else {
        size_t globalWorkSize[1] = {(size_t)size};
        engine->setKernelArg(VEC_MULT_KERNEL, 0, CL_MEM_READ_WRITE, size * sizeof(data_t), ret_vec, CL_TRUE);
        engine->setKernelArg(VEC_MULT_KERNEL, 1, CL_MEM_READ_ONLY, size * sizeof(data_t), vec, CL_TRUE);
        engine->enqueueKernel(VEC_MULT_KERNEL, 1, globalWorkSize, NULL, true);
        engine->enqueueReadBuffer(VEC_MULT_KERNEL, 0, size * sizeof(data_t), ret_vec);
    }
}

void mat_vec_add(data_t* ret_mat, const data_t* vec, int row, int col, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                ret_mat[i * col + j] += vec[j];
            }
        }
    } else {
        size_t globalWorkSize[2] = {(size_t)col, (size_t)row};
        engine->setKernelArg(MAT_VEC_ADD_KERNEL, 0, CL_MEM_READ_WRITE, row * col * sizeof(data_t), ret_mat, CL_TRUE);
        engine->setKernelArg(MAT_VEC_ADD_KERNEL, 1, CL_MEM_READ_ONLY, col * sizeof(data_t), vec, CL_TRUE);
        engine->setKernelArg(MAT_VEC_ADD_KERNEL, 2, 0, sizeof(int), &col);
        engine->enqueueKernel(MAT_VEC_ADD_KERNEL, 2, globalWorkSize, NULL, true);
        engine->enqueueReadBuffer(MAT_VEC_ADD_KERNEL, 0, row * col * sizeof(data_t), ret_mat);
    }
}

void mat_trans(data_t* ret_mat, const data_t* src_mat, int row, int col, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                ret_mat[j * row + i] = src_mat[i * col + j];
            }
        }
    } else {
        size_t globalWorkSize[2] = {(size_t)row, (size_t)col};
        engine->setKernelArg(MAT_TRANS_KERNEL, 0, CL_MEM_READ_WRITE, row * col * sizeof(data_t), ret_mat, CL_TRUE);
        engine->setKernelArg(MAT_TRANS_KERNEL, 1, CL_MEM_READ_ONLY, row * col * sizeof(data_t), src_mat, CL_TRUE);
        engine->setKernelArg(MAT_TRANS_KERNEL, 2, 0, sizeof(int), &row);
        engine->setKernelArg(MAT_TRANS_KERNEL, 3, 0, sizeof(int), &col);
        engine->enqueueKernel(MAT_TRANS_KERNEL, 2, globalWorkSize, NULL, true);
        engine->enqueueReadBuffer(MAT_TRANS_KERNEL, 0, row * col * sizeof(data_t), ret_mat);
    }
}

void mat_mult_col_cpu(data_t* ret_val, const data_t* a, const data_t* b, int row_a, int row_b, int col)
{
    data_t cell;
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < row_b; j++) {
            cell = 0.0;
            for (int k = 0; k < col; k++) {
                cell += a[i * col + k] * b[j * col + k];
            }
            ret_val[i * row_b + j] = cell;
        }
    }
}

void mat_mult_ocl(data_t* ret_val, const data_t* a, const data_t* b, int row_a, int row_b, int col_b, OclWorkEngine* engine)
{
    size_t globalWorkSize[2] = {(size_t)col_b, (size_t)row_a};
    engine->setKernelArg(MAT_MULT_KERNEL, 0, CL_MEM_READ_WRITE, row_a * col_b * sizeof(data_t), ret_val, CL_TRUE);
    engine->setKernelArg(MAT_MULT_KERNEL, 1, CL_MEM_READ_ONLY, row_a * row_b * sizeof(data_t), a, CL_TRUE);
    engine->setKernelArg(MAT_MULT_KERNEL, 2, CL_MEM_READ_ONLY, row_b * col_b * sizeof(data_t), b, CL_TRUE);
    engine->setKernelArg(MAT_MULT_KERNEL, 3, 0, sizeof(int), &row_b);
    engine->setKernelArg(MAT_MULT_KERNEL, 4, 0, sizeof(int), &col_b);
    engine->enqueueKernel(MAT_MULT_KERNEL, 2, globalWorkSize, NULL, true);
    engine->enqueueReadBuffer(MAT_MULT_KERNEL, 0, row_a * col_b * sizeof(data_t), ret_val);
}

void mat_mult_col(data_t* ret_val, const data_t* a, const data_t* b, int row_a, int row_b, int col, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        mat_mult_col_cpu(ret_val, a, b, row_a, row_b, col);
    } else {
        data_t* b_T = new data_t[row_b * col];
        mat_trans(b_T, b, row_b, col, engine);
        mat_mult_ocl(ret_val, a, b_T, row_a, col, row_b, engine);
        delete [] b_T;
    }
}

void mat_mult(data_t* ret_val, const data_t* a, const data_t* b, int row_a, int row_b, int col_b, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        data_t* b_T = new data_t[row_b * col_b];
        mat_trans(b_T, b, row_b, col_b);
        mat_mult_col(ret_val, a, b_T, row_a, col_b, row_b);
        delete [] b_T;
    } else {
        mat_mult_ocl(ret_val, a, b, row_a, row_b, col_b, engine);
    }
}

void mat_mult_row(data_t* ret_val, const data_t* a, const data_t* b, int col_a, int col_b, int row, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        data_t* a_T = new data_t[row * col_a];
        data_t* b_T = new data_t[row * col_b];
        mat_trans(a_T, a, row, col_a);
        mat_trans(b_T, b, row, col_b);
        mat_mult_col(ret_val, a_T, b_T, col_a, col_b, row);
        delete [] b_T;
        delete [] a_T;
    } else {
        data_t* a_T = new data_t[row * col_a];
        mat_trans(a_T, a, row, col_a, engine);
        mat_mult(ret_val, a_T, b, col_a, row, col_b, engine);
        delete [] a_T;
    }
}

// ---------------------- Member functions -------------------------------

// ---------------------- Functions for loading dataset -------------------------------

void Network::load_train_file(const char* file_name)
{
    if (worker_rank == MPI_MASTER) {
        std::ifstream input(file_name);
        if (input.is_open()) {
            for (int i = 0; i < train_data_size; i++) {
                unsigned char byte;
                input.read((char*)&byte, 1);
                for (int j = 0; j < sizes[num_layers-1]; j++) {
                    train_data_y[i * sizes[num_layers-1] + j] = 0.0;
                }
                train_data_y[i * sizes[num_layers-1] + (int)byte] = 1.0;
                for (int j = 0; j < sizes[0]; j++) {
                    input.read((char*)&byte, 1);
                    train_data_x[i * sizes[0] + j] = (data_t)byte / (1 << 8);
                }
            }
            input.close();
        } else {
            std::cout << "Failed to open file " << file_name << std::endl;
        }
    }
}

void Network::load_test_file(const char* file_name)
{
    if (worker_rank == MPI_MASTER) {
        std::ifstream input(file_name);
        if (input.is_open()) {
            for (int i = 0; i < test_data_size; i++) {
                unsigned char byte;
                input.read((char*)&byte, 1);
                test_data_y[i] = (data_t)byte;
                for (int j = 0; j < sizes[0]; j++) {
                    input.read((char*)&byte, 1);
                    test_data_x[i * sizes[0] + j] = (data_t)byte / (1 << 8);
                }
            }
            input.close();
        } else {
            std::cout << "Failed to open file " << file_name << std::endl;
        }
    }
}

void Network::load_dataset(const char *train_file_name, const char *test_file_name, int train_data_size, int test_data_size)
{
    this->train_data_size = train_data_size;
    this->test_data_size = test_data_size;
    
    train_data_x = new data_t[train_data_size * sizes[0]];
    train_data_y = new data_t[train_data_size * sizes[num_layers-1]];
    test_data_x = new data_t[test_data_size * sizes[0]];
    test_data_y = new data_t[test_data_size];
    
    NanoTimer timer_load;
    timer_load.restart();
    if (worker_rank == MPI_MASTER) {
        load_train_file(train_file_name);
        load_test_file(test_file_name);
    }
    timer_load.stop();
    
    NanoTimer timer_bcast;
    timer_bcast.restart();
    if (useMPI) {
        MPI_Bcast(train_data_x, train_data_size * sizes[0], MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
        MPI_Bcast(train_data_y, train_data_size * sizes[num_layers-1], MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
        MPI_Bcast(test_data_x, test_data_size * sizes[0], MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
        MPI_Bcast(test_data_y, test_data_size, MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
    }
    timer_bcast.stop();
    
    if (worker_rank == MPI_MASTER) {
        *debug_out << "Successfully loaded all train and test data on root node (ms): " << timer_load.getElapsedMS() << std::endl;
        *debug_out << "MPI bcast dataset (ms): " << timer_bcast.getElapsedMS() << std::endl;
    }
}

Network::Network(const int* sizes, int num_layers, bool useOCL, bool useMPI, std::ostream* out)
{
    this->useMPI = useMPI;
    if (useMPI) {
        MPI_Comm_size(MPI_COMM_WORLD, &worker_count);
        MPI_Comm_rank(MPI_COMM_WORLD, &worker_rank);
    }
    
    this->useOCL = useOCL;
    if (useOCL) {
        int index = 0;
        if (useMPI) {
            // Balance load for nodes that have multiple OpenCL devices (e.g. NewRiver has 2 GPUs per node)
            char* host_name = new char[MPI_MAX_PROCESSOR_NAME];
            int len;
            char* host_names = NULL;
            int* lens = NULL;
            MPI_Get_processor_name(host_name, &len);
            if (worker_rank == MPI_MASTER) {
                host_names = new char[MPI_MAX_PROCESSOR_NAME * worker_count];
                lens = new int[worker_count];
            }
            MPI_Gather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_MASTER, MPI_COMM_WORLD);
            MPI_Gather(&len, 1, MPI_INT, lens, 1, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);
            std::map<std::string, int> nameToIndex;
            int* indices = NULL;
            if (worker_rank == MPI_MASTER) {
                indices = new int[worker_count];
                char* curr_name = host_names;
                for (int i = 0; i < worker_count; i++) {
                    std::string name(curr_name, lens[i]);
                    if (nameToIndex.find(name) == nameToIndex.end()) {
                        nameToIndex[name] = 0;
                    } else {
                        nameToIndex[name]++;
                    }
                    indices[i] = nameToIndex[name];
                }
            }
            MPI_Scatter(indices, 1, MPI_INT, &index, 1, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);
            if (indices != NULL) {
                delete [] indices;
            }
            if (lens != NULL) {
                delete [] lens;
            }
            if (host_names != NULL) {
                delete [] host_names;
            }
            delete [] host_name;
        }
        this->engine = new OclWorkEngine(index);
        this->engine->createProgram(OCL_PROGRAM);
        this->engine->createKernel(SIGMOID_KERNEL);
        this->engine->createKernel(SIGMOID_PRIME_KERNEL);
        this->engine->createKernel(VEC_MULT_KERNEL);
        this->engine->createKernel(MAT_VEC_ADD_KERNEL);
        this->engine->createKernel(MAT_TRANS_KERNEL);
        this->engine->createKernel(MAT_MULT_KERNEL);
    }
    
    if (out) {
        debug_out = out;
    }
    
    this->num_layers = num_layers;
    
    this->sizes = new int[num_layers];
    for (int i = 0; i < num_layers; i++) {
        this->sizes[i] = sizes[i];
    }
    
    biases = new data_t*[num_layers-1];
    for (int i = 0; i < num_layers - 1; i++) {
        biases[i] = new data_t[sizes[i+1]];
    }
    
    weights = new data_t*[num_layers-1];
    for (int i = 0; i < num_layers - 1; i++) {
        weights[i] = new data_t[sizes[i+1] * sizes[i]];
    }
    
    nabla_w = new data_t*[num_layers-1];
    for (int i = 0; i < num_layers - 1; i++) {
        nabla_w[i] = new data_t[sizes[i+1] * sizes[i]];
    }
    
    if (useMPI) {
        nabla_w_sum = new data_t*[num_layers-1];
        for (int i = 0; i < num_layers - 1; i++) {
            nabla_w_sum[i] = new data_t[sizes[i+1] * sizes[i]];
        }
    } else {
        nabla_w_sum = nabla_w;
    }
    
    if (worker_rank == MPI_MASTER) {
        // Initialize weights and biases as a standard normal distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<data_t> d(0, 1);
        
        for (int i = 0; i < num_layers - 1; i++) {
            for (int j = 0; j < sizes[i+1]; j++) {
                biases[i][j] = d(gen);
            }
        }

        for (int i = 0; i < num_layers - 1; i++) {
            for (int j = 0; j < sizes[i+1] * sizes[i]; j++) {
                weights[i][j] = d(gen) / sqrt(sizes[i]);
            }
        }
    }
}

Network::~Network()
{
    if (useOCL) {
        delete engine;
    }
    
    clean_network();
    
    if (nabla_w_sum != NULL) {
        for (int i = 0; i < num_layers - 1; i++) {
            delete [] nabla_w_sum[i];
        }
        delete [] nabla_w_sum;
    }
    
    if (nabla_w != NULL) {
        for (int i = 0; i < num_layers - 1; i++) {
            delete [] nabla_w[i];
        }
        delete [] nabla_w;
    }
    
    for (int i = 0; i < num_layers - 1; i++) {
        delete [] weights[i];
    }
    delete [] weights;
    
    for (int i = 0; i < num_layers - 1; i++) {
        delete [] biases[i];
    }
    delete [] biases;
    
    delete [] sizes;
    
    if (train_data_x != NULL) {
        delete [] train_data_x;
    }
    if (train_data_y != NULL) {
        delete [] train_data_y;
    }
    if (test_data_x != NULL) {
        delete [] test_data_x;
    }
    if (test_data_y != NULL) {
        delete [] test_data_y;
    }
}

void Network::clean_network()
{
    if (primes != NULL) {
        for (int i = 0; i < num_layers - 1; i++) {
            delete [] primes[i];
        }
        delete [] primes;
    }
    
    if (activations != NULL) {
        for (int i = 0; i < num_layers; i++) {
            delete [] activations[i];
        }
        delete [] activations;
    }
    
    if (nabla_b != NULL) {
        for (int i = 0; i < num_layers - 1; i++) {
            delete [] nabla_b[i];
        }
        delete [] nabla_b;
    }
    
    if (nabla_b_sum != NULL) {
        for (int i = 0; i < num_layers - 1; i++) {
            delete [] nabla_b_sum[i];
        }
        delete [] nabla_b_sum;
    }
}

void Network::SGD(int epochs, int mini_batch_size, data_t eta, data_t lmbda)
{
    NanoTimer timer_epoch;
    NanoTimer timer_shuffle;
    NanoTimer timer_mpi_shuffle_scatter;
    NanoTimer timer_copy;
    NanoTimer timer_bcast_train;
    NanoTimer timer_backprop;
    NanoTimer timer_reduce_gradients;
    NanoTimer timer_update;
    NanoTimer timer_bcast_eval;
    NanoTimer timer_worker_eval;
    NanoTimer timer_reduce_eval;
    
    this->mini_batch_size = mini_batch_size;
    this->worker_batch_size = mini_batch_size / worker_count;
    
    clean_network();
    
    nabla_b = new data_t*[num_layers-1];
    for (int i = 0; i < num_layers - 1; i++) {
        nabla_b[i] = new data_t[worker_batch_size * sizes[i+1]];
    }
    
    if (useMPI) {
        nabla_b_sum = new data_t*[num_layers-1];
        for (int i = 0; i < num_layers - 1; i++) {
            nabla_b_sum[i] = new data_t[worker_batch_size * sizes[i+1]];
        }
    } else {
        nabla_b_sum = nabla_b;
    }
    
    activations = new data_t*[num_layers];
    for (int i = 0; i < num_layers; i++) {
        activations[i] = new data_t[worker_batch_size * sizes[i]];
    }
    
    primes = new data_t*[num_layers-1];
    for (int i = 0; i < num_layers - 1; i++) {
        primes[i] = new data_t[worker_batch_size * sizes[i+1]];
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    int* all_idx = NULL;
    if (worker_rank == MPI_MASTER) {
        all_idx = new int[train_data_size];
        for (int i = 0; i < train_data_size; i++) {
            all_idx[i] = i;
        }
    }
    
    worker_train_size = train_data_size / worker_count;
    worker_test_size = test_data_size / worker_count;
    
    int* idx;
    if (useMPI) {
        idx = new int[worker_train_size];
    } else {
        idx = all_idx;
    }
    
    int num_iterations = train_data_size / mini_batch_size;
    data_t* mini_batch_y = new data_t[worker_batch_size * sizes[num_layers-1]];
    
    for (int i = 0; i < epochs; i++) {
        timer_epoch.restart();
        
        timer_shuffle.restart();
        if (worker_rank == MPI_MASTER) {
            shuffle(&all_idx[0], &all_idx[train_data_size-1], gen);
        }
        timer_shuffle.stop();
        
        if (useMPI) {
            timer_mpi_shuffle_scatter.restart();
            // MPI Scatter all_idx --> idx
            MPI_Scatter(all_idx, worker_train_size, MPI_INT, idx, worker_train_size, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);
            timer_mpi_shuffle_scatter.stop();
        }
        
        timer_copy.restart(); timer_copy.stop();
        timer_bcast_train.restart(); timer_bcast_train.stop();
        timer_backprop.restart(); timer_backprop.stop();
        timer_reduce_gradients.restart(); timer_reduce_gradients.stop();
        timer_update.restart(); timer_update.stop();
        
        for (int j = 0; j < num_iterations; j++) {
            timer_copy.resume();
            for (int k = 0; k < worker_batch_size; k++) {
                for (int l = 0; l < sizes[0]; l++) {
                    activations[0][k * sizes[0] + l] = train_data_x[idx[j * worker_batch_size + k] * sizes[0] + l];
                }
            }
            for (int k = 0; k < worker_batch_size; k++) {
                for (int l = 0; l < sizes[num_layers-1]; l++) {
                    mini_batch_y[k * sizes[num_layers-1] + l] = train_data_y[idx[j * worker_batch_size + k] * sizes[num_layers-1] + l];
                }
            }
            timer_copy.stop();
            
            timer_bcast_train.resume();
            bcast_network();
            timer_bcast_train.stop();
            
            timer_backprop.resume();
            backprop(mini_batch_y);
            timer_backprop.stop();
            
            timer_reduce_gradients.resume();
            reduce_gradients();
            timer_reduce_gradients.stop();
            
            timer_update.resume();
            update_mini_batch(eta, lmbda);
            timer_update.stop();
        }
        
        timer_bcast_eval.restart();
        bcast_network();
        timer_bcast_eval.stop();
        
        timer_worker_eval.restart();
        int num_correct = evaluate(&test_data_x[worker_rank * worker_test_size * sizes[0]], &test_data_y[worker_rank * worker_test_size]);
        timer_worker_eval.stop();
        
        timer_reduce_eval.restart();
        if (useMPI) {
            int worker_correct = num_correct;
            MPI_Reduce(&worker_correct, &num_correct, 1, MPI_INT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);
        }
        timer_reduce_eval.stop();
        
        timer_epoch.stop();
        
        if (worker_rank == MPI_MASTER) {
            *debug_out << std::endl << "Epoch " << i << std::endl;
            *debug_out << "Random shuffle (ms):        " << timer_shuffle.getElapsedMS() << std::endl;
            *debug_out << "MPI scatter indices (ms):   " << timer_mpi_shuffle_scatter.getElapsedMS() << std::endl;
            *debug_out << "Copy shuffled data (ms):    " << timer_copy.getElapsedMS() << std::endl;
            *debug_out << "MPI bcast train (ms):       " << timer_bcast_train.getElapsedMS() << std::endl;
            *debug_out << "Back propagation (ms):      " << timer_backprop.getElapsedMS() << std::endl;
            *debug_out << "MPI reduce gradients (ms):  " << timer_reduce_gradients.getElapsedMS() << std::endl;
            *debug_out << "Update w & b (ms):          " << timer_update.getElapsedMS() << std::endl;
            *debug_out << "MPI bcast evaluation (ms):  " << timer_bcast_eval.getElapsedMS() << std::endl;
            *debug_out << "Evaluation (ms):            " << timer_worker_eval.getElapsedMS() << std::endl;
            *debug_out << "MPI reduce evaluation (ms): " << timer_reduce_eval.getElapsedMS() << std::endl;
            *debug_out << "Total (ms):                 " << timer_epoch.getElapsedMS() << std::endl;
            *debug_out << "Test accuracy: " << num_correct << " / " << test_data_size << std::endl;
        }
    }
    
    delete [] mini_batch_y;
    if (useMPI) {
        delete [] idx;
    }
    delete [] all_idx;
}

void Network::bcast_network()
{
    if (useMPI) {
        // MPI: broadcast network weights & biases to all workers
        for (int i = 0; i < num_layers - 1; i++) {
            MPI_Bcast(biases[i], sizes[i+1], MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
        }
        for (int i = 0; i < num_layers - 1; i++) {
            MPI_Bcast(weights[i], sizes[i+1] * sizes[i], MPI_FLOAT, MPI_MASTER, MPI_COMM_WORLD);
        }
    }
}

void Network::reduce_gradients()
{
    if (useMPI) {
        // MPI Reduce nabla_w & nabla_b
        for (int i = 0; i < num_layers - 1; i++) {
            MPI_Reduce(nabla_w[i], nabla_w_sum[i], sizes[i+1] * sizes[i], MPI_FLOAT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);
        }
        for (int i = 0; i < num_layers - 1; i++) {
            MPI_Reduce(nabla_b[i], nabla_b_sum[i], worker_batch_size * sizes[i+1], MPI_FLOAT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);
        }
    }
}

void Network::update_mini_batch(data_t eta, data_t lmbda)
{
    if (worker_rank == MPI_MASTER) {
        for (int j = 0; j < worker_batch_size; j++) {
            for (int i = 0; i < num_layers - 1; i++) {
                for (int k = 0; k < sizes[i+1]; k++) {
                    biases[i][k] = biases[i][k] - (eta / worker_batch_size) * nabla_b_sum[i][j * sizes[i+1] + k];
                }
            }
        }
        data_t reg = 1.0 - eta * lmbda / train_data_size;
        for (int i = 0; i < num_layers - 1; i++) {
            for (int j = 0; j < sizes[i+1] * sizes[i]; j++) {
                weights[i][j] = reg * weights[i][j] - (eta / worker_batch_size) * nabla_w_sum[i][j];
            }
        }
    }
}

void Network::feedforward()
{
    for (int i = 0; i < num_layers - 1; i++) {
        mat_mult_col(activations[i+1], activations[i], weights[i], worker_batch_size, sizes[i+1], sizes[i], engine);
        mat_vec_add(activations[i+1], biases[i], worker_batch_size, sizes[i+1], engine);
        sigmoid_prime(primes[i], activations[i+1], worker_batch_size * sizes[i+1], engine);
        sigmoid(activations[i+1], worker_batch_size * sizes[i+1], engine);
    }
}

void Network::cost_derivative(const data_t *y)
{
    for (int i = 0; i < worker_batch_size * sizes[num_layers-1]; i++) {
        nabla_b[num_layers-2][i] = activations[num_layers-1][i] - y[i];
    }
}

void Network::backprop(const data_t *y)
{
    feedforward();
    cost_derivative(y);
    mat_mult_row(nabla_w[num_layers-2], nabla_b[num_layers-2], activations[num_layers-2], sizes[num_layers-1], sizes[num_layers-2], worker_batch_size, engine);
    for (int i = num_layers - 3; i >= 0; i--) {
        mat_mult(nabla_b[i], nabla_b[i+1], weights[i+1], worker_batch_size, sizes[i+2], sizes[i+1], engine);
        vec_mult(nabla_b[i], primes[i], worker_batch_size * sizes[i+1], engine);
        mat_mult_row(nabla_w[i], nabla_b[i], activations[i], sizes[i+1], sizes[i], worker_batch_size, engine);
    }
}

int Network::evaluate(const data_t *test_data_x, const data_t *test_data_y)
{
    int num_iterations = worker_test_size / worker_batch_size;
    int arg_max;
    data_t* cell;
    int sum = 0;
    for (int i = 0; i < num_iterations; i++) {
        for (int j = 0; j < worker_batch_size * sizes[0]; j++) {
            activations[0][j] = test_data_x[i * worker_batch_size * sizes[0] + j];
        }
        feedforward();
        for (int j = 0; j < worker_batch_size; j++) {
            arg_max = 0;
            cell = &activations[num_layers-1][j * sizes[num_layers-1]];
            for (int k = 1; k < sizes[num_layers-1]; k++) {
                if (cell[k] > cell[arg_max]) {
                    arg_max = k;
                }
            }
            if (arg_max == (int)test_data_y[i * worker_batch_size + j]) {
                sum++;
            }
        }
    }
    return sum;
}
