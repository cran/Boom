#ifndef BOOM_RINTERFACE_CHECK_INTERRUPT_H_
#define BOOM_RINTERFACE_CHECK_INTERRUPT_H_

namespace BOOM {

// Utility to check to see if an interrupt signal was called.  This is
// useful to get around the issue of R's error handling not allowing C++ to
// exit gracefully.  Intended to be used with the RErrorReporter, which will
// call R's internal error handling after the current function exits.
//
// WARNING: THIS WILL ALSO EAT ALL OTHER ERRORS IN THE EVENT LOOP AS WELL. If
// you are depending on certain errors to do certain things you must handle
// that on your own.
//
// Code based on post by Simon Urbanek:
// https://stat.ethz.ch/pipermail/r-devel/2011-April/060702.html
//
// Example code:
//
// SEXP my_function_called_from_r(SEXP args) {
//   RErrorReporter errors;
//   ...
//   while(true) {
//     ...
//     if(RCheckInterrupt()) {
//       # Do any necessary cleanup here
//       errors.SetError(error_message);
//       return R_NilValue;
//    }
// }
// ...

bool RCheckInterrupt();

}  // namespace BOOM

#endif  // BOOM_RINTERFACE_CHECK_INTERRUPT_H_
