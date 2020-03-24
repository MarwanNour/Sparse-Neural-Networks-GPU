
#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
    float elapsedTime;
} Timer;

static void startTime(Timer* timer) {
        gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
    
    timer->elapsedTime = ((float) ((timer->endTime.tv_sec - timer->startTime.tv_sec)
                        + (timer->endTime.tv_usec - timer->startTime.tv_usec)/1.0e6));
}

static void printElapsedTime(Timer timer, const char* label) {
    printf("%s: %f s\n", label, timer.elapsedTime);
}

static void stopTimeAndPrint(Timer* timer, const char* label) {
    stopTime(timer);
    printElapsedTime(*timer, label);
}

static void stopTimeAndPrintWithRate(Timer* timer, const char* timeLabel, const char* rateLabel, unsigned int units) {
    stopTime(timer);
    float rate = units / timer->elapsedTime;
    printf("%s: %f s (%f %s/s)\n", timeLabel, timer->elapsedTime, rate, rateLabel);
}

#endif

