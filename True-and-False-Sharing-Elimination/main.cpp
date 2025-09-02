#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>
#include <chrono>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;

// Max different files
const int MAX_FILES = 10;
const int MAX_SIZE = 10;
int MAX_THREADS = 5; // You can update this

struct t_data {
  uint32_t tid;
};

struct padded_uint64_t {
  alignas(64) uint64_t count;
};


// struct to keep track of the number of occurrences of a word
struct word_tracker {
  padded_uint64_t word_count[5];
  uint64_t total_lines_processed;
  uint64_t total_words_processed;
  pthread_mutex_t final_update_mutex;
} tracker;

// Shared queue, to be read by producers
std::queue<std::string> shared_pq;
// updates to shared queue
pthread_mutex_t pq_mutex = PTHREAD_MUTEX_INITIALIZER;

// lock var to update to total line counter
pthread_mutex_t line_count_mutex = PTHREAD_MUTEX_INITIALIZER;

// each thread read a file and put the tokens in line into std out
void* thread_runner(void*);

void print_usage(char* prog_name) {
  cerr << "usage: " << prog_name << " <producer count> <input file>\n";
  exit(EXIT_FAILURE);
}

void print_counters() {
  for (int id = 0; id < MAX_THREADS; ++id) {
    std::cout << "Thread " << id << " counter: " << tracker.word_count[id].count
              << '\n';
  }
}

void fill_producer_buffer(std::string& input) {
  std::fstream input_file;
  input_file.open(input, ios::in);
  if (!input_file.is_open()) {
    cerr << "Error opening the top-level input file!" << endl;
    exit(EXIT_FAILURE);
  }

  std::filesystem::path p(input);
  std::string line;
  while (getline(input_file, line)) {
    shared_pq.push(p.parent_path() / line);
  }
}

int thread_count = 0;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    print_usage(argv[0]);
  }

  thread_count = strtol(argv[1], NULL, 10);
  MAX_THREADS = thread_count;
  std::string input = argv[2];
  fill_producer_buffer(input);

  pthread_t threads_worker[thread_count];

  int file_count;

  struct t_data* args_array =
      (struct t_data*)malloc(sizeof(struct t_data) * thread_count);
  for (int i = 0; i < thread_count; i++)
    tracker.word_count[i].count = 0;
  tracker.total_lines_processed = 0;
  tracker.total_words_processed = 0;
  tracker.final_update_mutex = PTHREAD_MUTEX_INITIALIZER;


  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < thread_count; i++) {
    args_array[i].tid = i;
    pthread_create(&threads_worker[i], nullptr, thread_runner,
                   (void*)&args_array[i]);
  }

  for (int i = 0; i < thread_count; i++)
    pthread_join(threads_worker[i], NULL);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
  cout << "Total Time Taken: " << duration.count() << "ms\n";

  print_counters();
  cout << "Total words processed: " << tracker.total_words_processed << "\n";
  cout << "Total line processed: " << tracker.total_lines_processed << "\n";

  return EXIT_SUCCESS;
}

void* thread_runner(void* th_args) {
  struct t_data* args = (struct t_data*)th_args;
  uint32_t thread_id = args->tid;
  std::fstream input_file;
  std::string fileName;
  std::string line;

  uint64_t local_line_count = 0;
  uint64_t local_word_count = 0;

  pthread_mutex_lock(&pq_mutex);
  if (!shared_pq.empty()) {
    fileName = shared_pq.front();
    shared_pq.pop();
  } else {
    pthread_mutex_unlock(&pq_mutex);
    pthread_exit(nullptr);
  }
  pthread_mutex_unlock(&pq_mutex);

  input_file.open(fileName.c_str(), ios::in);
  if (!input_file.is_open()) {
    cerr << "Error opening input file from a thread!" << endl;
    exit(EXIT_FAILURE);
  }

  while (getline(input_file, line)) {
    local_line_count++;
    std::string delimiter = " ";
    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      local_word_count++;
      token = line.substr(0, pos);
      line.erase(0, pos + delimiter.length());
    }
    if (!line.empty()) {
      local_word_count++;
    }
  }

  input_file.close();

  tracker.word_count[thread_id].count = local_word_count;

  pthread_mutex_lock(&tracker.final_update_mutex);
  tracker.total_lines_processed += local_line_count;
  tracker.total_words_processed += local_word_count;
  pthread_mutex_unlock(&tracker.final_update_mutex);

  pthread_exit(nullptr);
}
