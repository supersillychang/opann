//
//  main.cpp
//  oclengine
//
//  Created by Chang Sun on 1/7/18.
//  Copyright © 2018 Chang Sun. All rights reserved.
//

#include <iostream>
#include <cstring>

#include "oclworkengine.h"
#include "nanotimer.h"

typedef float data_t;

#define OCL_PROGRAM             "./kernels.cl"
#define SIGMOID_KERNEL          "sigmoid"
#define SIGMOID_PRIME_KERNEL    "sigmoid_prime"
#define VEC_MULT_KERNEL         "vec_mult"
#define MAT_VEC_ADD_KERNEL      "mat_vec_add"
#define MAT_TRANS_KERNEL        "mat_trans"
#define MAT_MULT_COL_KERNEL     "mat_mult_col"
#define MAT_MULT_ROW_KERNEL     "mat_mult_row"
#define MAT_MULT_KERNEL         "mat_mult"

NanoTimer timer_seq, timer_setup, timer_mem_create, timer_kernel_comp, timer_copy_back;
bool isWarmUp;

cl_ulong elapsed;
cl_ulong profiling_memory_create_time;
cl_ulong profiling_kernel_comp_time;
cl_ulong profiling_copy_back_time;

void sigmoid(data_t* ret_val, int size, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        if (!isWarmUp) {
            timer_seq.resume();
        }
        for (int i = 0; i < size; i++) {
            ret_val[i] = 1.0 / (1.0 + exp(-ret_val[i]));
        }
        if (!isWarmUp) {
            timer_seq.stop();
        }
    } else {
        size_t globalWorkSize[1] = {(size_t)size};
        if (!isWarmUp) {
            timer_mem_create.resume();
        }
        elapsed = engine->setKernelArg(SIGMOID_KERNEL, 0, CL_MEM_READ_WRITE, size * sizeof(data_t), ret_val, CL_TRUE);
        if (!isWarmUp) {
            timer_mem_create.stop();
            profiling_memory_create_time += elapsed;
            timer_kernel_comp.resume();
        }
        elapsed = engine->enqueueKernel(SIGMOID_KERNEL, 1, globalWorkSize, NULL, true);
        if (!isWarmUp) {
            timer_kernel_comp.stop();
            profiling_kernel_comp_time += elapsed;
            timer_copy_back.resume();
        }
        elapsed = engine->enqueueReadBuffer(SIGMOID_KERNEL, 0, size * sizeof(data_t), ret_val);
        if (!isWarmUp) {
            timer_copy_back.stop();
            profiling_copy_back_time += elapsed;
        }
    }
}

void sigmoid_prime(data_t* ret_val, const data_t* z, int size, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        if (!isWarmUp) {
            timer_seq.resume();
        }
        data_t sigmoid;
        for (int i = 0; i < size; i++) {
            sigmoid = 1.0 / (1.0 + exp(-z[i]));
            ret_val[i] = sigmoid * (1.0 - sigmoid);
        }
        if (!isWarmUp) {
            timer_seq.stop();
        }
    } else {
        size_t globalWorkSize[1] = {(size_t)size};
        if (!isWarmUp) {
            timer_mem_create.resume();
        }
        elapsed = engine->setKernelArg(SIGMOID_PRIME_KERNEL, 0, CL_MEM_READ_WRITE, size * sizeof(data_t), ret_val, CL_TRUE);
        elapsed += engine->setKernelArg(SIGMOID_PRIME_KERNEL, 1, CL_MEM_READ_ONLY, size * sizeof(data_t), z, CL_TRUE);
        if (!isWarmUp) {
            timer_mem_create.stop();
            profiling_memory_create_time += elapsed;
            timer_kernel_comp.resume();
        }
        elapsed = engine->enqueueKernel(SIGMOID_PRIME_KERNEL, 1, globalWorkSize, NULL, true);
        if (!isWarmUp) {
            timer_kernel_comp.stop();
            profiling_kernel_comp_time += elapsed;
            timer_copy_back.resume();
        }
        elapsed = engine->enqueueReadBuffer(SIGMOID_PRIME_KERNEL, 0, size * sizeof(data_t), ret_val);
        if (!isWarmUp) {
            timer_copy_back.stop();
            profiling_copy_back_time += elapsed;
        }
    }
}

void vec_mult(data_t* ret_vec, const data_t* vec, int size, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        if (!isWarmUp) {
            timer_seq.resume();
        }
        for (int i = 0; i < size; i++) {
            ret_vec[i] *= vec[i];
        }
        if (!isWarmUp) {
            timer_seq.stop();
        }
    } else {
        size_t globalWorkSize[1] = {(size_t)size};
        if (!isWarmUp) {
            timer_mem_create.resume();
        }
        elapsed = engine->setKernelArg(VEC_MULT_KERNEL, 0, CL_MEM_READ_WRITE, size * sizeof(data_t), ret_vec, CL_TRUE);
        elapsed += engine->setKernelArg(VEC_MULT_KERNEL, 1, CL_MEM_READ_ONLY, size * sizeof(data_t), vec, CL_TRUE);
        if (!isWarmUp) {
            timer_mem_create.stop();
            profiling_memory_create_time += elapsed;
            timer_kernel_comp.resume();
        }
        elapsed = engine->enqueueKernel(VEC_MULT_KERNEL, 1, globalWorkSize, NULL, true);
        if (!isWarmUp) {
            timer_kernel_comp.stop();
            profiling_kernel_comp_time += elapsed;
            timer_copy_back.resume();
        }
        elapsed = engine->enqueueReadBuffer(VEC_MULT_KERNEL, 0, size * sizeof(data_t), ret_vec);
        if (!isWarmUp) {
            timer_copy_back.stop();
            profiling_copy_back_time += elapsed;
        }
    }
}

void mat_vec_add(data_t* ret_mat, const data_t* vec, int row, int col, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        if (!isWarmUp) {
            timer_seq.resume();
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                ret_mat[i * col + j] += vec[j];
            }
        }
        if (!isWarmUp) {
            timer_seq.stop();
        }
    } else {
        size_t globalWorkSize[2] = {(size_t)col, (size_t)row};
        if (!isWarmUp) {
            timer_mem_create.resume();
        }
        elapsed = engine->setKernelArg(MAT_VEC_ADD_KERNEL, 0, CL_MEM_READ_WRITE, row * col * sizeof(data_t), ret_mat, CL_TRUE);
        elapsed += engine->setKernelArg(MAT_VEC_ADD_KERNEL, 1, CL_MEM_READ_ONLY, col * sizeof(data_t), vec, CL_TRUE);
        elapsed += engine->setKernelArg(MAT_VEC_ADD_KERNEL, 2, 0, sizeof(int), &col);
        if (!isWarmUp) {
            timer_mem_create.stop();
            profiling_memory_create_time += elapsed;
            timer_kernel_comp.resume();
        }
        elapsed = engine->enqueueKernel(MAT_VEC_ADD_KERNEL, 2, globalWorkSize, NULL, true);
        if (!isWarmUp) {
            timer_kernel_comp.stop();
            profiling_kernel_comp_time += elapsed;
            timer_copy_back.resume();
        }
        elapsed = engine->enqueueReadBuffer(MAT_VEC_ADD_KERNEL, 0, row * col * sizeof(data_t), ret_mat);
        if (!isWarmUp) {
            timer_copy_back.stop();
            profiling_copy_back_time += elapsed;
        }
    }
}

void mat_trans(data_t* ret_mat, const data_t* src_mat, int row, int col, OclWorkEngine* engine = NULL)
{
    if (engine == NULL) {
        if (!isWarmUp) {
            timer_seq.resume();
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                ret_mat[j * row + i] = src_mat[i * col + j];
            }
        }
    } else {
        size_t globalWorkSize[2] = {(size_t)row, (size_t)col};
        if (!isWarmUp) {
            timer_mem_create.resume();
        }
        elapsed = engine->setKernelArg(MAT_TRANS_KERNEL, 0, CL_MEM_READ_WRITE, row * col * sizeof(data_t), ret_mat, CL_TRUE);
        elapsed += engine->setKernelArg(MAT_TRANS_KERNEL, 1, CL_MEM_READ_ONLY, row * col * sizeof(data_t), src_mat, CL_TRUE);
        elapsed += engine->setKernelArg(MAT_TRANS_KERNEL, 2, 0, sizeof(int), &row);
        elapsed += engine->setKernelArg(MAT_TRANS_KERNEL, 3, 0, sizeof(int), &col);
        if (!isWarmUp) {
            timer_mem_create.stop();
            profiling_memory_create_time += elapsed;
            timer_kernel_comp.resume();
        }
        elapsed = engine->enqueueKernel(MAT_TRANS_KERNEL, 2, globalWorkSize, NULL, true);
        if (!isWarmUp) {
            timer_kernel_comp.stop();
            profiling_kernel_comp_time += elapsed;
            timer_copy_back.resume();
        }
        elapsed = engine->enqueueReadBuffer(MAT_TRANS_KERNEL, 0, row * col * sizeof(data_t), ret_mat);
        if (!isWarmUp) {
            timer_copy_back.stop();
            profiling_copy_back_time += elapsed;
        }
    }
}

void mat_mult_col_cpu(data_t* ret_val, const data_t* a, const data_t* b, int row_a, int row_b, int col)
{
    if (!isWarmUp) {
        timer_seq.resume();
    }
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
    if (!isWarmUp) {
        timer_seq.stop();
    }
}

void mat_mult_ocl(data_t* ret_val, const data_t* a, const data_t* b, int row_a, int row_b, int col_b, OclWorkEngine* engine)
{
    size_t globalWorkSize[2] = {(size_t)col_b, (size_t)row_a};
    if (!isWarmUp) {
        timer_mem_create.resume();
    }
    elapsed = engine->setKernelArg(MAT_MULT_KERNEL, 0, CL_MEM_READ_WRITE, row_a * col_b * sizeof(data_t), ret_val, CL_TRUE);
    elapsed += engine->setKernelArg(MAT_MULT_KERNEL, 1, CL_MEM_READ_ONLY, row_a * row_b * sizeof(data_t), a, CL_TRUE);
    elapsed += engine->setKernelArg(MAT_MULT_KERNEL, 2, CL_MEM_READ_ONLY, row_b * col_b * sizeof(data_t), b, CL_TRUE);
    elapsed += engine->setKernelArg(MAT_MULT_KERNEL, 3, 0, sizeof(int), &row_b);
    elapsed += engine->setKernelArg(MAT_MULT_KERNEL, 4, 0, sizeof(int), &col_b);
    if (!isWarmUp) {
        timer_mem_create.stop();
        profiling_memory_create_time += elapsed;
        timer_kernel_comp.resume();
    }
    elapsed = engine->enqueueKernel(MAT_MULT_KERNEL, 2, globalWorkSize, NULL, true);
    if (!isWarmUp) {
        timer_kernel_comp.stop();
        profiling_kernel_comp_time += elapsed;
        timer_copy_back.resume();
    }
    elapsed = engine->enqueueReadBuffer(MAT_MULT_KERNEL, 0, row_a * col_b * sizeof(data_t), ret_val);
    if (!isWarmUp) {
        timer_copy_back.stop();
        profiling_copy_back_time += elapsed;
    }
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


int main(int argc, const char * argv[]) {
    
    int row_a = atoi(argv[2]);
    int row_b = atoi(argv[3]);
    int col_b = atoi(argv[4]);

    int trials = atoi(argv[5]);
    int ocl_iters = atoi(argv[6]);
    int seq_iters = ocl_iters / 2;
    
    std::cout << "Kernel: " << argv[1] << std::endl;
    std::cout << "row_a: " << row_a << std::endl;
    std::cout << "row_b: " << row_b << std::endl;
    std::cout << "col_b: " << col_b << std::endl;
    
    int size_a = 0;
    int size_b = 0;
    int size_c = 0;
    
    if (strcmp(argv[1], SIGMOID_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_c = size_a;
        std::cout << "Problem size: [" << size_a << "]" << std::endl;
    } else if (strcmp(argv[1], SIGMOID_PRIME_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_c = size_a;
        std::cout << "Problem size: [" << size_a << "]" << std::endl;
    } else if (strcmp(argv[1], VEC_MULT_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_b = size_a;
        size_c = size_a;
        std::cout << "Problem size: [" << row_b << ", " << col_b << "] .* [" << row_b << ", " << col_b << "]" << std::endl;
    } else if (strcmp(argv[1], MAT_VEC_ADD_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_b = col_b;
        size_c = size_a;
        std::cout << "Problem size: [" << row_b << ", " << col_b << "] + [" << col_b << "]" << std::endl;
    } else if (strcmp(argv[1], MAT_TRANS_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_c = size_a;
        std::cout << "Problem size: [" << row_b << ", " << col_b << "] + [" << col_b << "]" << std::endl;
    } else if (strcmp(argv[1], MAT_MULT_COL_KERNEL) == 0) {
        size_a = row_b * row_a;
        size_b = col_b * row_a;
        size_c = row_b * col_b;
        std::cout << "Problem size: [" << row_b << ", " << row_a << "] * [" << row_a << ", " << col_b << "]" << std::endl;
    } else if (strcmp(argv[1], MAT_MULT_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_b = col_b * row_a;
        size_c = row_b * row_a;
        std::cout << "Problem size: [" << row_b << ", " << col_b << "] * [" << col_b << ", " << row_a << "]" << std::endl;
    } else if (strcmp(argv[1], MAT_MULT_ROW_KERNEL) == 0) {
        size_a = row_b * col_b;
        size_b = row_b * row_a;
        size_c = col_b * row_a;
        std::cout << "Problem size: [" << col_b << ", " << row_b << "] * [" << row_b << ", " << row_a << "]" << std::endl;
    }
    
    int warmup = 10;
    
    int threshold = 1048576; // 2^20
    
    if (size_a >= threshold || size_b >= threshold || size_c >= threshold) {
        seq_iters = 1;
        ocl_iters = 1;
        warmup = 1;
    }
    
    std::cout << std::endl;
    
    data_t* a = new data_t[size_a];
    for (int i = 0; i < size_a; i++) {
        a[i] = 0.1f * i;
    }
    
    data_t* b = NULL;
    if (size_b > 0) {
        b = new data_t[size_b];
        for (int i = 0; i < size_b; i++) {
            b[i] = 0.1f * i;
        }
    }
    
    data_t* c = new data_t[size_c];
    
    data_t* result = new data_t[size_c];
    
    uint64_t sec, nano;
    
    for (int t = -1; t < trials; t++) {
        
        isWarmUp = false;
        
        timer_seq.restart(); timer_seq.stop();
        
        for (int i = 0; i < seq_iters + warmup; i++) {
            
            isWarmUp = (i < warmup);
            
            if (strcmp(argv[1], SIGMOID_KERNEL) == 0) {
                for (int j = 0; j < size_a; j++) {
                    c[j] = a[j];
                }
                sigmoid(c, size_a);
            } else if (strcmp(argv[1], SIGMOID_PRIME_KERNEL) == 0) {
                sigmoid_prime(c, a, size_a);
            } else if (strcmp(argv[1], MAT_MULT_COL_KERNEL) == 0) {
                mat_mult_col(c, a, b, row_b, col_b, row_a);
            } else if (strcmp(argv[1], MAT_TRANS_KERNEL) == 0) {
                mat_trans(c, a, row_b, col_b);
            } else if (strcmp(argv[1], MAT_MULT_ROW_KERNEL) == 0) {
                mat_mult_row(c, a, b, col_b, row_a, row_b);
            } else if (strcmp(argv[1], MAT_MULT_KERNEL) == 0) {
                mat_mult(c, a, b, row_b, col_b, row_a);
            } else if (strcmp(argv[1], MAT_VEC_ADD_KERNEL) == 0) {
                for (int j = 0; j < size_a; j++) {
                    c[j] = a[j];
                }
                mat_vec_add(c, b, row_b, col_b);
            } else if (strcmp(argv[1], VEC_MULT_KERNEL) == 0) {
                for (int j = 0; j < size_a; j++) {
                    c[j] = a[j];
                }
                vec_mult(c, b, size_a);
            }
        
        }
        
        profiling_memory_create_time = 0;
        profiling_kernel_comp_time = 0;
        profiling_copy_back_time = 0;
        
        timer_mem_create.restart(); timer_mem_create.stop();
        timer_kernel_comp.restart(); timer_kernel_comp.stop();
        timer_copy_back.restart(); timer_copy_back.stop();
        
        timer_setup.restart();
        OclWorkEngine *engine = new OclWorkEngine(0, true);
        engine->createProgram(OCL_PROGRAM);
        engine->createKernel(SIGMOID_KERNEL);
        engine->createKernel(SIGMOID_PRIME_KERNEL);
        engine->createKernel(VEC_MULT_KERNEL);
        engine->createKernel(MAT_VEC_ADD_KERNEL);
        engine->createKernel(MAT_TRANS_KERNEL);
        engine->createKernel(MAT_MULT_KERNEL);
        timer_setup.stop();
        
        for (int i = 0; i < ocl_iters + warmup; i++){
            
            isWarmUp = (i < warmup);
            
            if (strcmp(argv[1], SIGMOID_KERNEL) == 0) {
                for (int j = 0; j < size_a; j++) {
                    result[j] = a[j];
                }
                sigmoid(result, size_a, engine);
            } else if (strcmp(argv[1], SIGMOID_PRIME_KERNEL) == 0) {
                sigmoid_prime(result, a, size_a, engine);
            } else if (strcmp(argv[1], MAT_MULT_COL_KERNEL) == 0) {
                mat_mult_col(result, a, b, row_b, col_b, row_a, engine);
            } else if (strcmp(argv[1], MAT_TRANS_KERNEL) == 0) {
                mat_trans(result, a, row_b, col_b, engine);
            } else if (strcmp(argv[1], MAT_MULT_ROW_KERNEL) == 0) {
                mat_mult_row(result, a, b, col_b, row_a, row_b, engine);
            } else if (strcmp(argv[1], MAT_MULT_KERNEL) == 0) {
                mat_mult(result, a, b, row_b, col_b, row_a, engine);
            } else if (strcmp(argv[1], MAT_VEC_ADD_KERNEL) == 0) {
                for (int j = 0; j < size_a; j++) {
                    result[j] = a[j];
                }
                mat_vec_add(result, b, row_b, col_b, engine);
            } else if (strcmp(argv[1], VEC_MULT_KERNEL) == 0) {
                for (int j = 0; j < size_a; j++) {
                    result[j] = a[j];
                }
                vec_mult(result, b, size_a, engine);
            }
            
        }
        
        delete engine;
        
        if (t >= 0) {
            std::cout << "Trial " << t << std::endl;
            timer_seq.getElapsed(sec, nano);
            std::cout << "Sequential (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) / seq_iters << std::endl;
            timer_setup.getElapsed(sec, nano);
            std::cout << "Setup OpenCL device in (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) << std::endl;
            std::cout << "Create memory objects (profiling) in (ms): " << profiling_memory_create_time * 1e-6 / ocl_iters << std::endl;
            timer_mem_create.getElapsed(sec, nano);
            std::cout << "Create memory objects in (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) / ocl_iters << std::endl;
            std::cout << "Kernel execution (profiling) finished in (ms): " << profiling_kernel_comp_time * 1e-6 / ocl_iters << std::endl;
            timer_kernel_comp.getElapsed(sec, nano);
            std::cout << "Kernel execution finished in (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) / ocl_iters << std::endl;
            std::cout << "Buffer read command (profiling) finished in (ms): " << profiling_copy_back_time * 1e-6 / ocl_iters << std::endl;
            timer_copy_back.getElapsed(sec, nano);
            std::cout << "Buffer read command finished in (ms): " << (sec * 1.0 * 1e3 + nano * 1e-6) / ocl_iters << std::endl;
            std::cout << std::endl;
        }
        
    }
    
    // Verify output
    bool success = true;
    data_t tolerance = 1e-9;
    for (int i = 0; i < size_c; i++) {
        if (abs(result[i] - c[i]) / abs(c[i]) > tolerance) {
            std::cout << i << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Results verified." << std::endl;
    } else {
        std::cout << "Computation error." << std::endl;
    }
    
    if (result != NULL) {
        delete [] result;
    }
    if (a != NULL) {
        delete [] a;
    }
    if (b != NULL) {
        delete [] b;
    }
    if (c != NULL) {
        delete [] c;
    }
    
    return 0;
}
