//
//  oclworkengine.h
//  oclengine
//
//  Created by Chang Sun on 12/19/17.
//  Copyright © 2017 Chang Sun. All rights reserved.
//

#ifndef oclworkengine_h
#define oclworkengine_h

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <math.h>
#include <unordered_map>

class OclWorkEngine
{
public:
    OclWorkEngine(int index = 0, bool enableProfiling = false);
    void createProgram(const char* fileName);
    void createKernel(const char* kernelName);
    cl_ulong setKernelArg(const char* kernelName, int argIndex, cl_mem_flags memFlag, size_t size, const void* argValue,
                          cl_bool blocking = CL_FALSE);
    cl_ulong enqueueKernel(const char* kernelName, cl_uint workDim, const size_t* globalWorkSize, const size_t* localWorkSize,
                           bool blocking = false);
    cl_ulong enqueueReadBuffer(const char* kernelName, int argIndex, size_t size, void* result,
                               cl_bool blocking = CL_TRUE);
    ~OclWorkEngine();
    
private:
    bool useDebug = true;
    bool enableProfiling = false;
    
    cl_context context = NULL;
    cl_device_id device = NULL;
    cl_command_queue commandQueue = NULL;
    cl_program program = NULL;
    
    std::unordered_map<std::string, cl_kernel> kernels;
    std::unordered_map<cl_kernel, std::unordered_map<int, cl_mem>> memObjects;
    
    void debugOut(const char* errMsg);
    void createContext();
    void createCommandQueue(int index);
    void releaseKernels();
    void releaseMemObjects();
    void cleanUp();
    
    cl_ulong getEventTime(cl_event* event);
};

#endif /* oclworkengine_h */
