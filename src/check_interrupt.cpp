#include "r_interface/check_interrupt.h"
#include "Rinternals.h"

namespace BOOM {

// Wrapper for R_CheckUserInterrupt
static void check_interrupt_func(void *dummy) {
  R_CheckUserInterrupt();
}

bool RCheckInterrupt() {
  // Checking in this way will ensure that R_CheckUserInterrupt will not
  // longjmp out of the current context, so C++ can clean up correctly.
  return (!R_ToplevelExec(check_interrupt_func, NULL));
}

}  // namespace BOOM
