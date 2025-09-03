#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define N (1e6)
#define NUM_THREADS (8)

// Shared variables
alignas(64) uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);

// Abstract base class
class LockBase
{
public:
  // Pure virtual function
  virtual void acquire(uint16_t tid) = 0;
  virtual void release(uint16_t tid) = 0;
};

typedef struct thr_args
{
  uint16_t m_id;
  LockBase *m_lock;
} ThreadArgs;

/** Use pthread mutex to implement lock routines */
class PthreadMutex : public LockBase
{
private:
  pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

public:
  void acquire(uint16_t tid) override
  {
    pthread_mutex_lock(&lock);
  }
  void release(uint16_t tid) override
  {
    pthread_mutex_unlock(&lock);
  }
};

/**************
FILTER LOCK
***************/
class FilterLock : public LockBase
{
private:
  struct alignas(64) padded_int
  {
    int val;
  };

  volatile padded_int *level;
  volatile padded_int *victim;

public:
  FilterLock()
  {
    level = new volatile padded_int[NUM_THREADS];
    victim = new volatile padded_int[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
      level[i].val = -1;
      victim[i].val = -1;
    }
  }

  void acquire(uint16_t tid) override
  {
    for (int i = 1; i < NUM_THREADS; i++)
    {
      level[tid].val = i;
      asm volatile("mfence" ::: "memory");
      victim[i].val = tid;
      asm volatile("mfence" ::: "memory");

      while (true)
      {
        bool ishighlevel = false;
        for (int k = 0; k < NUM_THREADS; k++)
        {
          if (k == tid)
            continue;
          if (level[k].val >= i)
          {
            ishighlevel = true;
            break;
          }
        }
        if (!ishighlevel || victim[i].val != tid)
        {
          break;
        }
        //__asm__ __volatile__("pause");
        asm volatile("pause");
      }
    }
  }

  void release(uint16_t tid) override
  {
    level[tid].val = -1;
    asm volatile("mfence" ::: "memory");
  }

  ~FilterLock()
  {
    delete[] level;
    delete[] victim;
  }
};

/***********
BAKERY LOCK
************/
class BakeryLock : public LockBase
{
private:
  struct alignas(64) padded_bool
  {
    volatile bool val;
  };
  struct alignas(64) padded_int
  {
    volatile int val;
  };

  volatile uint64_t global_counter;
  padded_bool *choosing;
  padded_int *ticket;

public:
  BakeryLock()
  {
    global_counter = 0;
    choosing = new padded_bool[NUM_THREADS];
    ticket = new padded_int[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
      choosing[i].val = false;
      ticket[i].val = 0;
    }
  }

  void acquire(uint16_t tid) override
  {
    choosing[tid].val = true;
    asm volatile("mfence" ::: "memory");

    ticket[tid].val = ({
                        uint64_t old_val = 1;
                        asm volatile("lock xaddq %0, %1"
                                     : "+r"(old_val), "+m"(global_counter)
                                     :
                                     : "memory");
                        old_val;
                      }) +
                      1;
    asm volatile("mfence" ::: "memory");

    choosing[tid].val = false;
    asm volatile("mfence" ::: "memory");

    for (int i = 0; i < NUM_THREADS; i++)
    {
      if (i == tid)
        continue;
      while (choosing[i].val)
      {
        //__asm__ __volatile__("pause");
        asm volatile("pause");
      }
      while (ticket[i].val != 0 &&
             (ticket[i].val < ticket[tid].val ||
              (ticket[i].val == ticket[tid].val && i < tid)))
      {
        //__asm__ __volatile__("pause");
        asm volatile("pause");
      }
    }
  }

  void release(uint16_t tid) override
  {
    ticket[tid].val = 0;
    asm volatile("mfence" ::: "memory");
  }

  ~BakeryLock()
  {
    delete[] choosing;
    delete[] ticket;
  }
};

/*****************
LAMPORT FAST LOCK
*****************/
class LamportFastLock : public LockBase
{
private:
  struct alignas(64) padded_bool
  {
    volatile bool val;
  };
  alignas(64) volatile uint64_t fast_lock;
  alignas(64) volatile uint64_t slow_lock;
  padded_bool *trying;

public:
  LamportFastLock()
  {
    trying = new padded_bool[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
      trying[i].val = false;
    fast_lock = -1;
    slow_lock = -1;
  }

  void acquire(uint16_t tid) override
  {
  start:
    trying[tid].val = true;
    asm volatile("mfence" ::: "memory");
    fast_lock = tid;
    asm volatile("mfence" ::: "memory");

    while (slow_lock != -1)
    {
      trying[tid].val = false;
      while (slow_lock != -1)
      {
        asm volatile("pause");
      }
      goto start;
    }

    slow_lock = tid;
    asm volatile("mfence" ::: "memory");

    if (fast_lock != tid)
    {
      trying[tid].val = false;
      asm volatile("mfence" ::: "memory");

      for (uint16_t i = 0; i < NUM_THREADS; i++)
      {
        if (i == tid)
          continue;

        while (trying[i].val)
        {
          //__asm__ __volatile__("pause");
          asm volatile("pause");
        }
      }

      if (slow_lock != tid)
      {
        while (slow_lock != -1)
        {
          //__asm__ __volatile__("pause");
          asm volatile("pause");
        }
        goto start;
      }
    }
  }

  void release(uint16_t tid) override
  {
    slow_lock = -1;
    asm volatile("mfence" ::: "memory");
    trying[tid].val = false;
    asm volatile("mfence" ::: "memory");
  }

  ~LamportFastLock()
  {
    delete[] trying;
  }
};


/**********
SPIN LOCK
***********/
class SpinLock : public LockBase
{
private:
  volatile uint64_t flag;

public:
  SpinLock()
  {
    flag = 0;
  }

  void acquire(uint16_t tid) override
  {
    uint64_t expected;
    do {
      expected = 0;
      asm volatile (
        "lock cmpxchgq %2, %1"
        : "=a"(expected), "+m"(flag)
        : "r"(1ULL), "0"(expected)
        : "memory"
      );
      if (expected != 0) {
        asm volatile("pause");
      }
    } while(expected != 0);
  }

  void release(uint16_t tid) override
  {
    asm volatile("mfence" ::: "memory");
    flag = 0;
  }

  ~SpinLock() {}
};


/***********
TICKET LOCK
************/
class TicketLock : public LockBase
{
private:
  alignas(64) volatile uint64_t next_ticket;
  alignas(64) volatile uint64_t now_serving;

public:
  TicketLock()
  {
    next_ticket = 0;
    now_serving = 0;
  }

  void acquire(uint16_t tid) override
  {
    uint64_t my_ticket;
    asm volatile(
        "lock xaddq %0, %1"
        : "=r"(my_ticket), "+m"(next_ticket)
        : "0"(1ULL)
        : "memory");

    asm volatile("mfence" ::: "memory");
    while (__atomic_load_n(&now_serving, __ATOMIC_ACQUIRE) != my_ticket)
    {
      //__asm__ __volatile__("pause");
      asm volatile("pause");
    }
  }

  void release(uint16_t tid) override
  {
    __atomic_add_fetch(&now_serving, 1, __ATOMIC_RELEASE);
    asm volatile("mfence" ::: "memory");
  }

  ~TicketLock() {}
};

/****************
ARRAY QUEUE LOCK
*****************/
class ArrayQLock : public LockBase
{
private:
  struct alignas(64) padded_bool
  {
    volatile bool val;
  };
  struct alignas(64) padded_uint64_t
  {
    volatile uint64_t val;
  };

  alignas(64) volatile uint64_t tail;
  padded_bool *flags;
  padded_uint64_t *my_slot;

public:
  ArrayQLock()
  {
    tail = 0;
    flags = new padded_bool[NUM_THREADS+1];
    flags[0].val = true;
    for (int i = 1; i < NUM_THREADS; i++)
      flags[i].val = false;
    my_slot = new padded_uint64_t[NUM_THREADS + 1];
  }

  void acquire(uint16_t tid) override
  {
    uint64_t myticket = ({
      uint64_t old = 1;
      asm volatile(
          "lock xaddq %0, %1"
          : "+r"(old), "+m"(tail)
          :
          : "memory");
      old;
    });

    my_slot[tid].val = myticket;
    myticket = myticket%NUM_THREADS;

    while (!flags[myticket].val)
    {
      // __asm__ __volatile__("pause");
      asm volatile("pause");
    }
  }

  void release(uint16_t tid) override
  {
    uint64_t myticket = my_slot[tid].val;
    myticket = myticket%NUM_THREADS;
    flags[myticket].val = false;
    asm volatile("mfence" ::: "memory");
    flags[(myticket+1)%NUM_THREADS].val = true;
  }

  ~ArrayQLock()
  {
    delete[] flags;
    delete[] my_slot;
  }
};

inline void critical_section()
{
  var1++;
  var2--;
}

/** Sync threads at the start to maximize contention */
pthread_barrier_t g_barrier;

void *thrBody(void *arguments)
{
  ThreadArgs *tmp = static_cast<ThreadArgs *>(arguments);
  if (false)
  {
    cout << "Thread id: " << tmp->m_id << " starting\n";
  }

  for (int i = 0; i < N; i++)
  {
    tmp->m_lock->acquire(tmp->m_id);
    critical_section();
    tmp->m_lock->release(tmp->m_id);
  }

  pthread_exit(NULL);
}

int main()
{
  int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
  if (error != 0)
  {
    cerr << "Error in barrier init.\n";
    exit(EXIT_FAILURE);
  }

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t tid[NUM_THREADS];
  ThreadArgs args[NUM_THREADS] = {{0}};

  // Pthread mutex
  LockBase *lock_obj = new PthreadMutex();
  HRTimer start = HR::now();
  uint16_t i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      cerr << "\nThread cannot be created : " << strerror(error) << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  void *status;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      cerr << "ERROR: return code from pthread_join() is " << error << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();

  assert(var1 == N * NUM_THREADS && var2 == 1);
  // cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Pthread mutex: Time taken (us): " << duration << "\n";
  delete lock_obj;

  // Filter lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);

  lock_obj = new FilterLock();
  start = HR::now();
  i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Filter lock: Time taken (us): " << duration << "\n";
  delete lock_obj;

  // Bakery lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);

  lock_obj = new BakeryLock();
  start = HR::now();
  i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Bakery lock: Time taken (us): " << duration << "\n";
  delete lock_obj;

  // Lamport fast lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);

  lock_obj = new LamportFastLock();
  start = HR::now();
  i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Lamport Fast Lock: Time taken (us): " << duration << "\n";
  delete lock_obj;

  // Spin lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);

  lock_obj = new SpinLock();
  start = HR::now();
  i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Spin lock: Time taken (us): " << duration << "\n";
  delete lock_obj;

  // Ticket lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);

  lock_obj = new TicketLock();
  start = HR::now();
  i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Ticket lock: Time taken (us): " << duration << "\n";
  delete lock_obj;

  // Array Q lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);

  lock_obj = new ArrayQLock();
  start = HR::now();
  i = 0;
  while (i < NUM_THREADS)
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
    if (error != 0)
    {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS)
  {
    error = pthread_join(tid[i], &status);
    if (error)
    {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Array Q lock: Time taken (us): " << duration << "\n";
  delete lock_obj;

  pthread_barrier_destroy(&g_barrier);
  pthread_attr_destroy(&attr);

  pthread_exit(NULL);
}

