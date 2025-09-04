## Cache Contention - True and False Sharing Elimination

### Performance Bug Analysis

#### Bug 1: High-Contention on Total Word Count (True Sharing)
Inside the innermost loop of `thread_runner`, the 
`tracker.word_count_mutex` is locked and unlocked for every single word processed. This serializes a significant portion 
of the parallel work. Threads spend more time waiting to acquire the 
lock than processing words, resulting in true sharing with high contention.

#### Bug 2: High-Contention on Total Line Count (True Sharing)
For every line a thread reads, it locks and unlocks the `line_count_mutex`, to increment the 
`tracker.total_lines_processed`. While less frequent than the per-word lock, this still introduces a major bottleneck.

#### Bug 3: Cache Thrashing on Per-Thread Word Counts (False Sharing)
The `tracker.word_count` array stores each thread's individual word count. Threads update their own element 
`tracker.word_count[thread.id]++`. The elements of the `word_count` array are contiguous in memory. A `uint64_t` is 8 bytes, so word-count[0], word-count[1], ..., word-count[4] will all reside on the same 64-byte cache line.  

When Thread 0 writes to word-count[0], it pulls the cache line into its core's cache in an "Exclusive" or "Modified" state. When Thread 1 then tries to write to word-count[1], it must invalidate Thread 0's copy and pull the line into its own cache. This back-and-forth invalidation, known as false sharing, creates massive memory bus traffic and stalls, even though the threads are writing to logically distinct variables.

---

### Modifications to Source Code

#### Eliminated True Sharing with Local Counters
To fix the high contention on global counters, each thread now uses its own private local variables (`local_line_count`, `local_word_count`). These local counters are incremented within the loops without requiring any locks. After a thread finishes processing its entire file, it acquires a single mutex `final_update_mutex` just once to add its local totals to the global `tracker.total_lines_processed` and `tracker.total_words_processed`. This reduces lock operations from one per-word/per-line to one per-thread.

#### Eliminated False Sharing with Padding
To solve the false sharing on the per-thread word-count array, a new struct `padded_uint64_t` was inserted. This struct uses `alignas(64)` to ensure that each count variable starts on a new 64-byte boundary. This forces each thread's counter onto a separate cache line, preventing false sharing. 

#### Corrected Word Counting Logic
A minor logic bug was fixed where the last word on each line was not being counted. The modified code now correctly increments the word count for the final token after the loop finishes. Also, when the number of threads is N, the word count array is modified to N, rather than fixed size of 4.

---

### Performance Gain Analysis

#### Performance comparison of initial vs. modified program

| Total lines | Total words | Initial time (ms) | Modified time (ms) | HITM (I) | HITM (M) |
|-------------|-------------|-------------------|--------------------|----------|----------|
| 500         | 5,964       | 15.8357           | 6.47798            | 9        | 0        |
| 500,000     | 6,130,000   | 556.26            | 331.825            | 1740     | 0        |

---

### Key Observations

- The modifications result in an improvement in performance and scalability, which is evident from both the execution timings and the perf c2c analysis.
- For number of files = 5, threads = 5, total lines = 500000 (1e5 lines/file), total words = 6130000 - Before Fix showed 3 hot shared cache lines with a total of 1740 LclHitm events, clearly indicating massive cache-level contention from both true and false sharing. After Fix, the "Global Shared Cache Line Event Information" section shows **Total Shared Cache Lines: 0**. The Shared Data Cache Line Table is empty. This proves that all cache-level contention has been eliminated with threads operating independently without interfering with one another.

- **Initial Program Performance** - The "Global Shared Cache Line Event Information" section shows 3 active shared cache lines. The most telling metric is the 1740 "Load Local HITM" events, which signifies that threads are frequently requesting cache lines that have been modified by other cores on the same CPU. Cache Line shows 60.29%, 32.99% and 6.72% of contention. The first one is the primary bottleneck and shows HITM events occurring at multiple offsets within the single cache line, showing false sharing.

- **Final Program Performance** - The "Global Shared Cache Line Event Information" now reports **Total Shared Cache Lines: 0**. The "Trace Event Information" shows **Load Local HITM: 0**. Shared Data Cache Line Table: This table is now empty. The absence of any shared cache line contention or HITM events proves that both the false sharing and true sharing problems have been eliminated. The padding fix (`alignas(64)`) successfully placed each per-thread counter on a separate cache line, stopping the false sharing. The use of thread-local counters successfully removed the high-frequency locking, eliminating the true sharing bottlenecks.

- **Execution Time Analysis** -  
The execution time measurements show a significant performance gain.  
Initial Program Runtime: **556.26 ms**  
Modified Program Runtime: **331.825 ms**  
The fix yields a **1.67x speedup**.
