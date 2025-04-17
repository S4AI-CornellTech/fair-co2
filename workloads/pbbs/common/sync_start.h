#include <time.h>
#include <unistd.h>

void sync_start(time_t start) {
  // sleep until the start time
  while(time(0) < start) {
    sleep(1);
  }
}