//
//  nanotimer.cpp
//  nanotimer
//
//  Created by Chang Sun on 10/16/17.
//  Copyright © 2017 Chang Sun. All rights reserved.
//

#include "nanotimer.h"

NanoTimer::NanoTimer()
{
    restart();
}

void NanoTimer::restart()
{
    struct timespec curr;
    clock_gettime(CLOCK_REALTIME, &curr);
    start.tv_sec = curr.tv_sec;
    start.tv_nsec = curr.tv_nsec;
    end.tv_sec = curr.tv_sec;
    end.tv_nsec = curr.tv_nsec;
    sec = 0;
    nano = 0;
}

void NanoTimer::stop()
{
    struct timespec curr;
    clock_gettime(CLOCK_REALTIME, &curr);
    end.tv_sec = curr.tv_sec;
    end.tv_nsec = curr.tv_nsec;
}

void NanoTimer::resume()
{
    uint64_t diff_sec;
    uint64_t diff_nano;
    getElapsed(diff_sec, diff_nano);
    restart();
    sec = diff_sec;
    nano = diff_nano;
}

void NanoTimer::getElapsed(uint64_t &sec, uint64_t &nano)
{
    uint64_t diff_sec = this->sec;
    uint64_t diff_nano = this->nano;
    diff_sec += end.tv_sec - start.tv_sec;
    if (end.tv_nsec < start.tv_nsec) {
        diff_sec += - 1;
        diff_nano += 1e9 + end.tv_nsec - start.tv_nsec;
    } else {
        diff_nano += end.tv_nsec - start.tv_nsec;
    }
    sec = diff_sec;
    nano = diff_nano;
}

uint64_t NanoTimer::getElapsedMS()
{
    uint64_t diff_sec;
    uint64_t diff_nano;
    getElapsed(diff_sec, diff_nano);
    return (diff_sec * 1e3 + diff_nano / 1e6);
}
