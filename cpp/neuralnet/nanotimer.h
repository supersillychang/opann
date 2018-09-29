//
//  nanotimer.h
//  nanotimer
//
//  Created by Chang Sun on 10/16/17.
//  Copyright © 2017 Chang Sun. All rights reserved.
//

#ifndef nanotimer_h
#define nanotimer_h

#include <time.h>
#include <cstdint>

class NanoTimer
{
public:
    NanoTimer();
    void restart();
    void stop();
    void resume();
    void getElapsed(uint64_t &sec, uint64_t &nano);
    uint64_t getElapsedMS();

private:
    struct timespec start;
    struct timespec end;
    uint64_t sec;
    uint64_t nano;
};

#endif /* nanotimer_h */
