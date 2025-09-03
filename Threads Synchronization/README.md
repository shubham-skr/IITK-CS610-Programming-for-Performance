## Synchronization

- **Shared Buffer:**
  - `buffer_mutex`: Ensures that only one thread, either a producer or a consumer thread, can write or read the shared buffer at any given time.
  - `buffer_not_full_condition`: Producers wait on this if the buffer is full. They are awakened by consumers who have removed data from the buffer.
  - `buffer_not_empty_condition`: Consumers wait on this if the buffer is empty. They are awakened by producers who have added data. 

- **Input File:**
  - `input_mutex`: Ensures that only one producer can read from the input file at a time. A thread locks it, reads its L lines, and then unlocks it, ensuring atomicity of the read operation.

- **Output File:**
  - `output_mutex`: Ensures that the block of lines retrieved by the consumer from the buffer is written to the output file contiguously without interruption from other consumer threads.

- **Producer Thread L-lines Atomicity:**
  - `producer_thread_turn_mutex`: Ensures that a producer writes its entire block of $L$ lines to the buffer atomically. This prevents other producers from interleaving writes, especially when $L > M$ and the thread must write in multiple chunks.

- **Atomic Variable:**
  - `producers_status`: This ensures that consumer threads only terminate when all producer threads have finished their work (reading all input and writing to the buffer is complete), and the buffer is empty.


<br>
<br>

## Compilation and Execution

### Compilation Command
```bash
g++ -std=c++17 synchronize.cpp -o synchronize.out -pthread
```

### Execution Command

```bash
The program is executed from the command line with six arguments:
1. The absolute path to the input file (R)
2. The number of producer threads (T)
3. The minimum lines to read (Lmin)
4. The maximum lines to read (Lmax)
5. The buffer size (M)
6. The path for the output file (W)

Example: ./synchronize.out /absolute/path/input.txt 4 5 10 20 /absolute/path/output.txt
```
