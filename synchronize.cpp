/******************************************

CS 610 Semester 2025–2026-I

Compilation command: 
With Makefile - make problem3
Without Makefile - g++ -std=c++17 -o synchronize.out synchronize.cpp -pthread

Execution command:
./synchronize.out <R:input_path> <T:producers> <Lmin:minimum_lines_read> \
<Lmax:maximum_lines_read> <M:buffer_lines> <W:output_path>

*******************************************/

#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <stdexcept>


/****************************
* Global Shared Resources
*****************************/ 

// Mutexes for synchronizing access to shared resources.
std::mutex input_mutex;      // Protects the input file read.
std::mutex output_mutex;     // Protects the output file write.
std::mutex buffer_mutex;     // Protects the shared buffer.
std::mutex producer_thread_turn_mutex;  // Protects producer's lines write atomicity. 

// Condition variables for coordination between producer and consumer threads.
std::condition_variable buffer_not_full_condition;  // Notifies producers when the buffer is not full.
std::condition_variable buffer_not_empty_condition; // Notifies consumers when the buffer is not empty.

// Shared buffer for writing and reading lines by producers and consumers.
std::vector<std::string> shared_buffer;
size_t shared_buffer_capacity; 

// Atomic status to signal all producers are done with their work.
std::atomic<bool> producers_status(false);

// Helper function to show status at console 
void show_status(const std::string& message) {
    std::cout << message << "\n";
}


/****************************
* Producer Thread Function
*****************************/

void producer(int id, std::ifstream& input_file, int lines_read_min, int lines_read_max) {
    // Random number generator this producer thread read lines
    std::random_device rd;
    std::mt19937 gen(rd() + id);
    std::uniform_int_distribution<> distrib(lines_read_min, lines_read_max);

    while(true) {
        std::vector<std::string> read_lines;

        // Critical section: Read from input file 
        {
            std::unique_lock<std::mutex> input_lock(input_mutex);
            
            if (!input_file.is_open() || input_file.peek() == EOF) {
                break; // Exit if file is closed or end of file reached
            }

            size_t lines_to_read = distrib(gen);
            std::string line;
            for(size_t i = 0; i < lines_to_read && std::getline(input_file, line); i++) {
                read_lines.push_back(line);
            }

            input_lock.unlock();

            if (read_lines.empty())
                break;
        }

        // Critical section: Write to shared buffer
        {
            std::unique_lock<std::mutex> producer_thread_turn_lock(producer_thread_turn_mutex);
            
            size_t lines_written = 0;
            while(lines_written < read_lines.size()) {
                std::unique_lock<std::mutex> buffer_lock(buffer_mutex);
                buffer_not_full_condition.wait(buffer_lock, [] {
                    return shared_buffer.size() < shared_buffer_capacity;
                });

                size_t available_buffer_space = shared_buffer_capacity - shared_buffer.size();
                size_t lines_to_write = std::min(available_buffer_space, read_lines.size() - lines_written);

                for(size_t i = 0; i < lines_to_write; i++) 
                    shared_buffer.push_back(std::move(read_lines[lines_written + i]));
                lines_written += lines_to_write;
                buffer_lock.unlock();
                buffer_not_empty_condition.notify_one();
            }
            producer_thread_turn_lock.unlock();
        }
    }
}


/****************************
* Consumer Thread Function
*****************************/

void consumer(std::ofstream& output_file) {
    while(true) {
        std::vector<std::string> lines_to_write;

        // Critical section: Read from shared buffer 
        {
            std::unique_lock<std::mutex> buffer_lock(buffer_mutex);
            buffer_not_empty_condition.wait(buffer_lock, [] {
                return !shared_buffer.empty() || producers_status.load();
            });

            if (shared_buffer.empty() && producers_status.load()) {
                break; 
            }

            lines_to_write.swap(shared_buffer);
            buffer_lock.unlock();
            buffer_not_full_condition.notify_all(); 
        }

        // Critical section: Write to output file
        {
            std::unique_lock<std::mutex> output_lock(output_mutex);
            for(const auto& line: lines_to_write) {
                output_file << line << "\n";
            }
            output_lock.unlock();
        }
    }
}


/****************************
* Main Function
*****************************/

int main(int argc, char* argv[]) {
    // Define arguments
    std::string R_input_file_path;
    int T_num_producers_threads;
    int Lmin_lines_read_min;
    int Lmax_lines_read_max;
    int M_buffer_capacity;
    std::string W_output_file_path;

    // Arguments parsing and validation
    try {
        if (argc != 7) 
            throw std::invalid_argument("Incorrect number of arguments provided.");

        R_input_file_path = argv[1];
        T_num_producers_threads = std::stoi(argv[2]);
        Lmin_lines_read_min = std::stoi(argv[3]);
        Lmax_lines_read_max = std::stoi(argv[4]);
        M_buffer_capacity = std::stoi(argv[5]);
        shared_buffer_capacity = M_buffer_capacity;
        W_output_file_path = argv[6];

        if (T_num_producers_threads <= 0)
            throw std::invalid_argument("T must be > 0");
        if (Lmin_lines_read_min <= 0) 
            throw std::invalid_argument("Lmin must be > 0");
        if (Lmax_lines_read_max < Lmin_lines_read_min)
            throw std::invalid_argument("Lmax must be >= Lmin");
        if (M_buffer_capacity <= 0) 
            throw std::invalid_argument("M must be > 0");

    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid arguments provided.\n" 
              << e.what() << "\n";
        std::cerr << "\nArguments: <R> <T> <Lmin> <Lmax> <M> <W>\n"
              << "  R:      Absolute path to the input file\n"
              << "  T:      Number of producer threads\n"
              << "  Lmin:   Minimum lines to read per turn\n"
              << "  Lmax:   Maximum lines to read per turn\n"
              << "  M:      Size of the shared buffer\n"
              << "  W:      Path to the output file\n";
        std::cerr << "\nPlease check constraints.\n"
              << "T must be > 0\n"
              << "Lmin must be > 0\n"
              << "Lmax must be >= Lmin\n"
              << "M must be > 0\n";

        return 1;
    }
    show_status("Arguments parsing and validation completed.");

    // Input file initialization
    std::ifstream input_file(R_input_file_path);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open the input file: " << R_input_file_path << "\n";
        return 1;
    }
    show_status("Input file initialized.");

    // Output file initialization
    std::ofstream output_file(W_output_file_path);
    if (!output_file.is_open()) {       
        std::cerr << "Error: Could not open the output file: " << W_output_file_path << "\n";
        return 1;
    }
    show_status("Output file initialized.");

    // Create threads for producers and consumers
    show_status("Starting producer and consumer threads...");
    show_status("Buffer capacity: " + std::to_string(shared_buffer_capacity));
    
    // Producer threads creation
    std::vector<std::thread> producers;
    for (int i = 0; i < T_num_producers_threads; ++i) 
        producers.emplace_back(producer, i+1, std::ref(input_file), Lmin_lines_read_min, Lmax_lines_read_max);
    show_status("Number of producer threads: " + std::to_string(producers.size()));

    // Consumer threads creation
    std::vector<std::thread> consumers;
    int num_consumers_threads = std::max(1, T_num_producers_threads / 2);
    for (int i = 0; i < num_consumers_threads; ++i) 
        consumers.emplace_back(consumer, std::ref(output_file));
    show_status("Number of consumer threads: " + std::to_string(consumers.size()));

    // Wait for all producer threads to finish
    for(auto& pt: producers) 
        pt.join();
    show_status("All producer threads have finished reading the input file.");

    // Signal that all producers are done
    producers_status.store(true);
    buffer_not_empty_condition.notify_all(); 

    // Wait for all consumer threads to finish
    for(auto& ct: consumers) 
        ct.join();
    show_status("All consumer threads have finished writing the output file.");

    show_status("Processing completed successfully.\nOutput written to: " + W_output_file_path);
    
    return 0;
}
