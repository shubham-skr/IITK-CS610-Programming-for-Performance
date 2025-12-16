#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <papi.h>

using std::cerr;
using std::cout;
using std::endl;
using std::uint64_t;

// Problem configuration
#define INP_H (1 << 6)
#define INP_W (1 << 6)
#define INP_D (1 << 6)
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)


// Helper function to handle PAPI errors
void handle_papi_error(int retval) {
    if (retval != PAPI_OK) {
        cerr << "PAPI error " << retval << ": " << PAPI_strerror(retval) << endl;
        exit(1);
    }
}


/**
 * Naive 3D convolution implementation
 * Cross-correlation without padding 
*/
void cc_3d_naive(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                      const uint64_t outputHeight, const uint64_t outputWidth,
                      const uint64_t outputDepth) {
    for (uint64_t i = 0; i < outputHeight; i++) {
        for (uint64_t j = 0; j < outputWidth; j++) {
            for (uint64_t k = 0; k < outputDepth; k++) {
                uint64_t sum = 0;
                for (uint64_t ki = 0; ki < FIL_H; ki++) {
                    for (uint64_t kj = 0; kj < FIL_W; kj++) {
                        for (uint64_t kk = 0; kk < FIL_D; kk++) {
                            sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D + (k + kk)] * kernel[ki][kj][kk];
                        }
                    }
                }
                result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
            }
        }
    }
}

/**
 * Blocked 3D convolution implementation
 * Cross-correlation without padding 
*/
void cc_3d_blocked(const uint64_t* input,
                   const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                   const uint64_t outputHeight, const uint64_t outputWidth,
                   const uint64_t outputDepth,
                   const uint64_t B_I, const uint64_t B_J, const uint64_t B_K) {
    for (uint64_t bi = 0; bi < outputHeight; bi += B_I) {
        for (uint64_t bj = 0; bj < outputWidth; bj += B_J) {
            for (uint64_t bk = 0; bk < outputDepth; bk += B_K) {
                for (uint64_t i = bi; i < std::min(bi + B_I, outputHeight); ++i) {
                    for (uint64_t j = bj; j < std::min(bj + B_J, outputWidth); ++j) {
                        for (uint64_t k = bk; k < std::min(bk + B_K, outputDepth); ++k) {
                            uint64_t sum = 0;
                            for (uint64_t ki = 0; ki < FIL_H; ki++) {
                                for (uint64_t kj = 0; kj < FIL_W; kj++) {
                                    for (uint64_t kk = 0; kk < FIL_D; kk++) {
                                        sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D + (k + kk)] * kernel[ki][kj][kk];
                                    }
                                }
                            }
                            result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
                        }
                    }
                }
            }
        }
    }
}


int main() {
    uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
    std::fill_n(input, INP_H * INP_W * INP_D, 1);
    uint64_t filter[FIL_H][FIL_W][FIL_D] = {{{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};
    uint64_t outputHeight = INP_H - FIL_H + 1;
    uint64_t outputWidth = INP_W - FIL_W + 1;
    uint64_t outputDepth = INP_D - FIL_D + 1;
    
    auto* result_naive = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
    auto* result_blocked = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
    
    const int num_runs = 5;

    // --- PAPI Initialization ---
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    int EventSet = PAPI_NULL;
    handle_papi_error(PAPI_create_eventset(&EventSet));
    
    // Add L1, L2, and L3 Data Cache Miss events
    int events[] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_TCM};
    handle_papi_error(PAPI_add_events(EventSet, events, 3));
    long long papi_values[3];
    
    // Benchmark Naive Implementation
    cout << "--- Benchmarking Naive Implementation ---" << endl;
    std::vector<double> naive_times;
    std::vector<long long> naive_l1_misses, naive_l2_misses, naive_l3_misses;

    for (int i = 0; i < num_runs; ++i) {
        std::fill_n(result_naive, outputHeight * outputWidth * outputDepth, 0);

        handle_papi_error(PAPI_start(EventSet));
        auto start = std::chrono::high_resolution_clock::now();
        
        cc_3d_naive(input, filter, result_naive, outputHeight, outputWidth, outputDepth);
        
        auto end = std::chrono::high_resolution_clock::now();
        handle_papi_error(PAPI_stop(EventSet, papi_values));

        std::chrono::duration<double, std::milli> duration = end - start;
        naive_times.push_back(duration.count());
        naive_l1_misses.push_back(papi_values[0]);
        naive_l2_misses.push_back(papi_values[1]);
        naive_l3_misses.push_back(papi_values[2]);
        handle_papi_error(PAPI_reset(EventSet));
    }

    double avg_naive_time = std::accumulate(naive_times.begin(), naive_times.end(), 0.0) / num_runs;
    long long avg_naive_l1_misses = std::accumulate(naive_l1_misses.begin(), naive_l1_misses.end(), 0LL) / num_runs;
    long long avg_naive_l2_misses = std::accumulate(naive_l2_misses.begin(), naive_l2_misses.end(), 0LL) / num_runs;
    long long avg_naive_l3_misses = std::accumulate(naive_l3_misses.begin(), naive_l3_misses.end(), 0LL) / num_runs;

    cout << "Average Naive Time: " << avg_naive_time << " ms\n";
    cout << "Average Naive L1 DCM: " << avg_naive_l1_misses << endl;
    cout << "Average Naive L2 DCM: " << avg_naive_l2_misses << endl;
    cout << "Average Naive L3 TCM: " << avg_naive_l3_misses << endl;

    // Benchmarking & Autotuning Blocked Implementation
    cout << "\n--- Benchmarking and Autotuning Blocked Implementation ---" << endl;
    std::vector<uint64_t> block_sizes_to_test = {4, 8, 16, 32, 64};
    uint64_t best_block_I = 0, best_block_J = 0, best_block_K = 0;
    double best_time = 1e18;
    long long best_l1_misses = 0, best_l2_misses = 0, best_l3_misses = 0;

    for (uint64_t b_size_i : block_sizes_to_test) {
        for(uint64_t b_size_j : block_sizes_to_test) {
            for(uint64_t b_size_k : block_sizes_to_test) {
                cout << "\nTesting Block Size (" << b_size_i << ", " << b_size_j << ", " << b_size_k << ")..." << endl;
                std::vector<double> current_times;
                std::vector<long long> current_l1_misses, current_l2_misses, current_l3_misses;

                for (int i = 0; i < num_runs; ++i) {
                    std::fill_n(result_blocked, outputHeight * outputWidth * outputDepth, 0);
                    
                    handle_papi_error(PAPI_start(EventSet));
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    cc_3d_blocked(input, filter, result_blocked, outputHeight, outputWidth, outputDepth, b_size_i, b_size_j, b_size_k);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    handle_papi_error(PAPI_stop(EventSet, papi_values));

                    std::chrono::duration<double, std::milli> duration = end - start;
                    current_times.push_back(duration.count());
                    current_l1_misses.push_back(papi_values[0]);
                    current_l2_misses.push_back(papi_values[1]);
                    current_l3_misses.push_back(papi_values[2]);
                    handle_papi_error(PAPI_reset(EventSet));
                }

                double avg_current_time = std::accumulate(current_times.begin(), current_times.end(), 0.0) / num_runs;
                long long avg_current_l1_misses = std::accumulate(current_l1_misses.begin(), current_l1_misses.end(), 0LL) / num_runs;
                long long avg_current_l2_misses = std::accumulate(current_l2_misses.begin(), current_l2_misses.end(), 0LL) / num_runs;
                long long avg_current_l3_misses = std::accumulate(current_l3_misses.begin(), current_l3_misses.end(), 0LL) / num_runs;

                cout << "  Average Blocked Time: " << avg_current_time << " ms\n";
                cout << "  Average Blocked L1 DCM: " << avg_current_l1_misses << endl;
                cout << "  Average Blocked L2 DCM: " << avg_current_l2_misses << endl;
                cout << "  Average Blocked L3 TCM: " << avg_current_l3_misses << endl;

                if (avg_current_time < best_time) {
                    best_time = avg_current_time;
                    best_block_I = b_size_i;
                    best_block_J = b_size_j;
                    best_block_K = b_size_k;
                    best_l1_misses = avg_current_l1_misses;
                    best_l2_misses = avg_current_l2_misses;
                    best_l3_misses = avg_current_l3_misses;
                }
    }
    
    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << "NAIVE IMPLEMENTATION:\n";
    std::cout << "  Average Time: " << avg_naive_time << " ms\n";
    std::cout << "  Average L1 DCM: " << avg_naive_l1_misses << "\n";
    std::cout << "  Average L2 DCM: " << avg_naive_l2_misses << "\n";
    std::cout << "  Average L3 TCM: " << avg_naive_l3_misses << "\n\n";

    std::cout << "BEST BLOCKED IMPLEMENTATION (Block size " << best_block_I << "x" << best_block_J << "x" << best_block_K << "):\n";
    std::cout << "  Average Time: " << best_time << " ms\n";
    std::cout << "  Average L1 DCM: " << best_l1_misses << "\n";
    std::cout << "  Average L2 DCM: " << best_l2_misses << "\n";
    std::cout << "  Average L3 TCM: " << best_l3_misses << "\n\n";

    std::cout << "Speedup: " << avg_naive_time / best_time << "x" << std::endl;

    // Cleanup PAPI
    handle_papi_error(PAPI_cleanup_eventset(EventSet));
    handle_papi_error(PAPI_destroy_eventset(&EventSet));
    
    // Free allocated memory
    delete[] input;
    delete[] result_naive;
    delete[] result_blocked;

    return 0;
}
