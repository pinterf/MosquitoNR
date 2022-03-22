//------------------------------------------------------------------------------
//		thread.cpp
//------------------------------------------------------------------------------

#include "mosquito_nr.h"

unsigned __stdcall RunThread(void* arg)
{
  ThreadInfo* th = (ThreadInfo*)arg;

  while (true) {
    WaitForSingleObject(th->job_start, INFINITE);
    if (th->close) break;
    (th->inst->*th->mt_func)(th->thread_id);
    SetEvent(th->job_finished);
  }

  _endthreadex(0);
  return 0;
}

MTInfo::MTInfo()
{
  threads = 0;

  for (int i = 0; i < MAX_THREADS; ++i) {
    th[i].job_start = NULL;
    th[i].job_finished = NULL;
    running[i] = NULL;
  }
}

MTInfo::~MTInfo()
{
  for (int i = 0; i < threads; ++i) if (running[i]) {
    th[i].close = true;
    SetEvent(th[i].job_start);
  }

  for (int i = 0; i < threads; ++i) if (running[i])
    WaitForSingleObject(running[i], INFINITE);

  for (int i = 0; i < threads; ++i) {
    if (th[i].job_start)    CloseHandle(th[i].job_start);
    if (th[i].job_finished) CloseHandle(th[i].job_finished);
    if (running[i])         CloseHandle(running[i]);
  }
}

bool MTInfo::CreateThreads(int _threads, MosquitoNR* inst)
{
  if (threads || _threads <= 0 || _threads > MAX_THREADS) return false;

  threads = _threads;

  for (int i = 0; i < threads; ++i) {
    th[i].inst = inst;
    th[i].thread_id = i;
    th[i].close = false;

    th[i].job_start = CreateEvent(NULL, FALSE, FALSE, NULL);
    th[i].job_finished = CreateEvent(NULL, FALSE, FALSE, NULL);

    if (!th[i].job_start || !th[i].job_finished || !(running[i] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &th[i], 0, NULL)))
      return false;
  }

  return true;
}

void MTInfo::ExecMTFunc(MTFunc mt_func)
{
  for (int i = 0; i < threads; ++i) {
    th[i].mt_func = mt_func;
    SetEvent(th[i].job_start);
  }

  for (int i = 0; i < threads; ++i)
    WaitForSingleObject(th[i].job_finished, INFINITE);
}
