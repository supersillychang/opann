//
//  main.cpp
//  neuralnet
//
//  Created by Chang Sun on 1/2/18.
//  Copyright © 2018 Chang Sun. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <mpi.h>
#include <string>
#include <cstring>

#include "network.h"
#include "nanotimer.h"


#define DATASET_MNIST                   0
#define DATASET_NAME_MNIST              "mnist"
#define TRAIN_DATA_MNIST                "../neuralnet_data/mnist_train.bin"
#define TEST_DATA_MNIST                 "../neuralnet_data/mnist_test.bin"
#define TRAIN_SIZE_MNIST                50000
#define TEST_SIZE_MNIST                 10000
#define INPUT_SIZE_MNIST                784
#define OUTPUT_SIZE_MNIST               10


#define DATASET_CIFAR10                 1
#define DATASET_NAME_CIFAR10            "cifar10"
#define TRAIN_DATA_CIFAR10              "../neuralnet_data/cifar10_train.bin"
#define TEST_DATA_CIFAR10               "../neuralnet_data/cifar10_test.bin"
#define TRAIN_SIZE_CIFAR10              50000
#define TEST_SIZE_CIFAR10               10000
#define INPUT_SIZE_CIFAR10              3072
#define OUTPUT_SIZE_CIFAR10             10


#define MAX_NUM_HIDDEN_LAYERS           100
#define DEFAULT_NUM_HIDDEN_LAYERS       1
#define DEFAULT_HIDDEN_LAYER_SIZE       32


#define DEFAULT_EPOCHS                  100
#define DEFAULT_MINI_BATCH_SIZE         1000
#define DEFAULT_LEARNING_ETA            0.5
#define DEFAULT_LEARNING_LMBDA          0.0


int main(int argc, const char * argv[]) {
    
    bool useOCL = false;
    bool useMPI = true;
    
    if (useMPI) {
        MPI_Init(NULL, NULL);
    }
    
    std::ostream* out = &std::cout;
    
    int sizes[MAX_NUM_HIDDEN_LAYERS];
    int num_layers = 1;
    
    int dataset = DATASET_MNIST;
    const char* train_file_name = TRAIN_DATA_MNIST;
    const char* test_file_name = TEST_DATA_MNIST;
    int train_data_size = TRAIN_SIZE_MNIST;
    int test_data_size = TEST_SIZE_MNIST;
    int input_layer_size = INPUT_SIZE_MNIST;
    int output_layer_size = OUTPUT_SIZE_MNIST;
    
    int epochs = DEFAULT_EPOCHS;
    int mini_batch_size = DEFAULT_MINI_BATCH_SIZE;
    data_t eta = DEFAULT_LEARNING_ETA;
    data_t lmbda = DEFAULT_LEARNING_LMBDA;
    
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-dataset") == 0 && i < argc - 1) {
            if (strcmp(argv[i+1], DATASET_NAME_CIFAR10) == 0) {
                dataset = DATASET_CIFAR10;
            }
        } else if (strcmp(argv[i], "-mpi") == 0) {
            useMPI = true;
        } else if (strcmp(argv[i], "-ocl") == 0) {
            useOCL = true;
        } else if (strcmp(argv[i], "-layers") == 0 && i < argc - 1) {
            int tmp;
            while (i < argc - 1 && (tmp = atoi(argv[++i]))) {
                sizes[num_layers++] = tmp;
            }
            i--;
        } else if (strcmp(argv[i], "-epochs") == 0 && i < argc - 1) {
            epochs = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "-batch_size") == 0 && i < argc - 1) {
            mini_batch_size = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "-eta") == 0 && i < argc - 1) {
            eta = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-lmbda") == 0 && i < argc - 1) {
            lmbda = atof(argv[i+1]);
        }
    }
    
    if (dataset == DATASET_CIFAR10) {
        train_file_name = TRAIN_DATA_CIFAR10;
        test_file_name = TEST_DATA_CIFAR10;
        train_data_size = TRAIN_SIZE_CIFAR10;
        test_data_size = TEST_SIZE_CIFAR10;
        input_layer_size = INPUT_SIZE_CIFAR10;
        output_layer_size = OUTPUT_SIZE_CIFAR10;
    }
    
    sizes[0] = input_layer_size;
    if (num_layers <= 1) {
        sizes[num_layers++] = DEFAULT_HIDDEN_LAYER_SIZE;
    }
    sizes[num_layers++] = output_layer_size;
    
    int world_rank = MPI_MASTER;
    if (useMPI) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    }
    if (world_rank == MPI_MASTER) {
        std::cout << "Number of layers: " << num_layers << std::endl;
        std::cout << "Network structure: ";
        for (int i = 0; i < num_layers; i++) {
            std::cout << sizes[i] << '\t';
        }
        std::cout << std::endl;
        
        if (useOCL) {
            std::cout << "Using OCL" << std::endl;
        }
        
        std::cout << "Dataset: ";
        if (dataset == DATASET_CIFAR10) {
            std::cout << DATASET_NAME_CIFAR10 << std::endl;
        } else {
            std::cout << DATASET_NAME_MNIST << std::endl;
        }
        
        std::cout << "Mini-batch size: ";
        std::cout << mini_batch_size << std::endl << std::endl;
    }
    
    Network net(sizes, num_layers, useOCL, useMPI, out);
    net.load_dataset(train_file_name, test_file_name, train_data_size, test_data_size);
    net.SGD(epochs, mini_batch_size, eta, lmbda);
    
    if (useMPI) {
        MPI_Finalize();
    }
    
    return 0;
}
