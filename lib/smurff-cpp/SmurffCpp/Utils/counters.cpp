/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <chrono>

#ifdef PROFILING

#include <cmath>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <cmath>

#include "counters.h"
#include "SmurffCpp/Utils/omp_util.h"

static std::mutex mtx;
static Counter *active_counter = 0;

Counter::Counter(std::string name)
    : name(name), diff(0), count(1), total_counter(false)
{
    mtx.lock();
    parent = active_counter;
    active_counter = this;
    mtx.unlock();

    fullname = (parent) ? parent->fullname + "/" + name : name; 

    start = tick();
}

Counter::Counter()
    : parent(0), name(std::string()), fullname(std::string()), diff(0), count(0), total_counter(true)
{
} 

Counter::~Counter() {
    if(total_counter) return;

    stop = tick();
    diff = stop - start;

    perf_data.local()[fullname] += *this;
    active_counter = parent;
}

void Counter::operator+=(const Counter &other) {
    if (name.empty()) 
    {
        name = other.name;
        fullname = other.fullname;
    }
    diff += other.diff;
    count += other.count;
}

std::string Counter::as_string(const Counter &total) const {
    std::ostringstream os;
    int percent = round(100.0 * diff / (total.diff + 0.000001));
    os << ">> " << fullname << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << "\t(" << percent << "%) in\t" << count << "\n";
    return os.str();
}

std::string Counter::as_string() const
{
    std::ostringstream os;
    os << ">> " << name << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << " in\t" << count << "\n";
    return os.str();
}

smurff::thread_vector<TotalsCounter> perf_data;

TotalsCounter::TotalsCounter(int p) : procid(p) {}

void TotalsCounter::print(int threadid) const {
    if (data.empty()) return;
    char hostname[1024];
    gethostname(hostname, 1024);
    std::cout << "\nTotals on " << hostname << " (" << procid << ") / thread " << threadid << ":\n";
    const auto total = data.find("main");
    for(auto &t : data)
        if (total != data.end())
            std::cout << t.second.as_string(total->second);
        else
            std::cout << t.second.as_string();
}

#endif // PROFILING

double tick() 
{
   return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
