struct GpuTimer {
    GpuTimer();           // cudaEventCreate both events
    ~GpuTimer();          // cudaEventDestroy both
    void start();         // cudaEventRecord(start_)
    void stop();          // cudaEventRecord(stop_) + cudaEventSynchronize(stop_)
    float elapsed_ms();   // cudaEventElapsedTime

private:
    cudaEvent_t start_, stop_;
};

inline GpuTimer::GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

inline GpuTimer::~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

inline void GpuTimer::start() {
    cudaEventRecord(start_);
}

inline void GpuTimer::stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
}

inline float GpuTimer::elapsed_ms() {
    float ms;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
}