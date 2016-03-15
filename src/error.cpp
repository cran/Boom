#include <stdarg.h>

#include "r_interface/error.h"
#include "Rinternals.h"

using std::string;

namespace BOOM {

RErrorReporter::~RErrorReporter() {
  if (HasError()) {
    // We need to call Rf_error with a dynamically allocated message.  To do
    // this, we create a SEXP to hold the message.  R will not UNPROTECT the
    // SEXP holding the message until some time after the message has been
    // copied into R's internal error buffer.
    SEXP s_error_message = PROTECT(Rf_mkChar(error_->c_str()));
    delete error_;

    Rf_error("%s", CHAR(s_error_message));
  }
}

void RErrorReporter::SetError(const string &error) {
  if (HasError()) {
    // Code should exit immediately once an error has been detected.
    // However, if for any reason we get multiple errors, keep the first one.
    return;
  }

  error_ = new string(error);
}

void RErrorReporter::SetError(const char *format, ...) {
  string error;
  va_list ap;
  va_start(ap, format);
  const int buf_size = 8192;
  char buf[buf_size];
  vsnprintf(buf, buf_size, format, ap);
  error = buf;

  SetError(error);
  va_end(ap);
}

const string &RErrorReporter::GetError() const {
  return *error_;
}

void RErrorReporter::Clear() {
  delete error_;
  error_ = NULL;
}

}  // namespace BOOM
