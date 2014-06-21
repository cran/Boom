#ifndef BOOM_RINTERFACE_ERROR_H_
#define BOOM_RINTERFACE_ERROR_H_

#include <string>

namespace BOOM {

// Utility class for reporting errors to R without resource leaks.
//
// Using this class is required as R's error handling code internally calls
// longjmp, which unwinds the stack without calling the C++ destructors.
// The class should be the first object allocated in the function called by R.
// To pass errors to R, code should call SetError() and immediately return from
// the function.  The object's destructor will automatically call R's error
// handling functions when the current function exits.
//
// Sample usage:
// SEXP my_function_called_from_r(SEXP args) {
//   RErrorReporter errors;
//   ...
//   if (some_condition) {
//     errors.SetError(error_message);
//     return R_NilValue;
//   }
//   ...
//   my_other_function(some_args, &error);
//   if (errors.HasError()) {
//     return R_NilValue;
//   }
//   ...

using std::string;


class RErrorReporter {
 public:
  RErrorReporter() { error_ = NULL; }

  // Calls Rf_error() if SetError() was called.
  ~RErrorReporter();

  // Sets the error.  When the object goes out of scope, will call Rf_Error()
  // with the passed in error message.
  // It is an error to call this function more than once on the object.  Once
  // an error has occured functions should exit immediately.
  void SetError(const string &error);

  void SetError(const char *format, ...);

  // TODO(kmillar): Add SetError(SEXP error).

  bool HasError() const { return error_ != NULL; }

  // For testing purposes, permit clearing the error message.  Thus,
  // the destructor will not invoke Rf_error.
  void Clear();

  // For testing purposes, permit inspecting the error message.
  // Requires: HasError().
  // Returns: The error message.
  const string &GetError() const;

 private:
  // An optional allocated string containing the error message.
  string *error_;

  // Disallow copying the object.
  RErrorReporter(const RErrorReporter &);
  const RErrorReporter &operator=(const RErrorReporter &);
};

}  // namespace BOOM

#endif  // BOOM_RINTERFACE_ERROR_H_
