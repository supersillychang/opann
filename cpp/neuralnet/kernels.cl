#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef float data_t;

__kernel void sigmoid(__global data_t* s)
{
    int gid = get_global_id(0);
    s[gid] = 1.0 / (1.0 + exp(-s[gid]));
}

__kernel void sigmoid_prime(__global data_t* sp,
                            __global const data_t* z)
{
    int gid = get_global_id(0);
    data_t s = 1.0 / (1.0 + exp(-z[gid]));
    sp[gid] = s * (1.0 - s);
}

__kernel void vec_mult(__global data_t* b,
                       __global const data_t* a)
{
    int gid = get_global_id(0);
    b[gid] *= a[gid];
}

__kernel void mat_vec_add(__global data_t* mat,
                          __global const data_t* vec,
                          int col)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    
    mat[gid_y * col + gid_x] += vec[gid_x];
}

__kernel void mat_trans(__global data_t* ret_mat,
                        __global const data_t* src_mat,
                        int row, int col)
{
    
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    
    ret_mat[gid_y * row + gid_x] = src_mat[gid_x * col + gid_y];
}

__kernel void mat_mult(__global data_t* c,
                       __global const data_t* a,
                       __global const data_t* b,
                       int row_b, int col_b)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    
    data_t value = 0;
    for (int k = 0; k < row_b; k++) {
        value += a[gid_y * row_b + k] * b[k * col_b + gid_x];
    }
    
    c[gid_y * col_b + gid_x] = value;
}
