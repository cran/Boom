/*
  Copyright (C) 2005-2011 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include <string>
#include <sstream>

#include <r_interface/boom_r_tools.hpp>
#include <cpputil/report_error.hpp>

#include <R.h>
#include <Rinternals.h>

namespace BOOM{
  SEXP getListElement(SEXP list, const std::string &name){
    SEXP elmt = R_NilValue;
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if(Rf_isNull(names)){
      report_error("attempt to use getListElement in a list with"
                   " no 'names' attribute.");
    }
    for(int i = 0; i < Rf_length(list); i++)
      if(name == CHAR(STRING_ELT(names, i))){
        elmt = VECTOR_ELT(list, i);
        break;
      }
    return elmt;
  }

  // Returns a vector of list names.  If an element does not have a
  // name then an empty string is created in place of the missing
  // name.
  std::vector<std::string> getListNames(SEXP list) {
    // There is no need to PROTECT list_names because they are an
    // attribute of list, and thus already protected.
    SEXP list_names = Rf_getAttrib(list, R_NamesSymbol);
    int n = Rf_length(list);
    if(list_names == R_NilValue){
      std::vector<std::string> ans(n, "");
      return ans;
    }
    std::vector<std::string> ans;
    ans.reserve(n);
    for(int i = 0; i < n; ++i){
      ans.push_back(CHAR(STRING_ELT(list_names, i)));
    }
    return ans;
  }

  // Sets the names attribute of list to a character vector equivalent
  // to 'names'.
  SEXP setListNames(SEXP list, const std::vector<std::string> &names) {
    int n = Rf_length(list);
    if(n != names.size()){
      report_error("'list' and 'names' are not the same size in setlistNames");
    }
    SEXP list_names;
    PROTECT(list_names = Rf_allocVector(STRSXP, n));
    for(int i = 0; i < n; ++i) {
      SET_STRING_ELT(list_names, i, Rf_mkChar(names[i].c_str()));
    }
    Rf_namesgets(list, list_names);
    UNPROTECT(1);
    return list;
  }

  std::vector<std::string> GetFactorLevels(SEXP r_factor) {
    SEXP r_factor_levels = Rf_getAttrib(r_factor, R_LevelsSymbol);
    return StringVector(r_factor_levels);
  }

  std::vector<std::string> StringVector(SEXP r_character_vector) {
    if (Rf_isNull(r_character_vector)) {
      return std::vector<std::string>();
    } else if (!Rf_isString(r_character_vector)) {
      report_error("StringVector expects a character vector argument");
    }
    int n = Rf_length(r_character_vector);
    std::vector<std::string> ans;
    ans.reserve(n);
    for(int i = 0; i < n; ++i){
      ans.push_back(CHAR(STRING_ELT(r_character_vector, i)));
    }
    return ans;
  }

  SEXP CharacterVector(const std::vector<std::string> & string_vector){
    SEXP ans;
    PROTECT(ans = Rf_allocVector(STRSXP, string_vector.size()));
    for(int i = 0; i < string_vector.size(); ++i){
      SET_STRING_ELT(ans, i, Rf_mkChar(string_vector[i].c_str()));
    }
    UNPROTECT(1);
    return ans;
  }

  SEXP appendListElement(SEXP list, SEXP new_element, const std::string &name){
    int n = Rf_length(list);
    SEXP ans;
    PROTECT(ans = Rf_allocVector(VECSXP, n+1));
    for(int i = 0; i < n; ++i){
      SET_VECTOR_ELT(ans, i, VECTOR_ELT(list, i));
    }
    SET_VECTOR_ELT(ans, n, new_element);

    SEXP old_list_names = Rf_getAttrib(list, R_NamesSymbol);
    SEXP list_names;
    PROTECT(list_names = Rf_allocVector(STRSXP, n+1));

    if(!Rf_isNull(old_list_names)){
      for(int i = 0; i < n; ++i){
        SET_STRING_ELT(list_names, i, STRING_ELT(old_list_names, i));
      }
    }
    SET_STRING_ELT(list_names, n, Rf_mkChar(name.c_str()));
    Rf_namesgets(ans, list_names);
    UNPROTECT(2);
    return ans;
  }

  SEXP appendListElements(SEXP r_list,
                          const std::vector<SEXP> &new_elements,
                          const std::vector<std::string> &new_element_names) {
    if (new_element_names.size() != new_elements.size()) {
      report_error("In appendListElements:  The vector of new elements must "
                   "be the same size as the vector of new element names.");
    }
    int original_list_length = Rf_length(r_list);
    SEXP ans;
    PROTECT(ans = Rf_allocVector(
        VECSXP, original_list_length + new_elements.size()));
    for (int i = 0; i < original_list_length; ++i) {
      SET_VECTOR_ELT(ans, i, VECTOR_ELT(r_list, i));
    }
    for (int i = 0; i < new_elements.size(); ++i) {
      SET_VECTOR_ELT(ans, i + original_list_length,
                     new_elements[i]);
    }
    std::vector<std::string> new_list_names = getListNames(r_list);
    for (int i = 0; i < new_element_names.size(); ++i) {
      new_list_names.push_back(new_element_names[i]);
    }
    ans = setListNames(ans, new_list_names);
    UNPROTECT(1);
    return ans;
  }

  SEXP CreateList(const std::vector<SEXP> &elements,
                  const std::vector<std::string> &element_names) {
    if (!element_names.empty()) {
      SEXP empty_list, ans;
      PROTECT(empty_list = Rf_allocVector(VECSXP, 0));
      PROTECT(ans = appendListElements(empty_list, elements, element_names));
      UNPROTECT(2);
      return(ans);
    } else {
      SEXP ans;
      PROTECT(ans = Rf_allocVector(VECSXP, elements.size()));
      for (int i = 0; i < elements.size(); ++i) {
        SET_VECTOR_ELT(ans, i, elements[i]);
      }
      UNPROTECT(1);
      return ans;
    }
  }

  std::vector<std::string> GetS3Class(SEXP object){
    SEXP rclass = Rf_getAttrib(object, R_ClassSymbol);
    return StringVector(rclass);
  }

  std::pair<int,int> GetMatrixDimensions(SEXP matrix){
    if(!Rf_isMatrix(matrix)){
      report_error("GetMatrixDimensions called on a non-matrix object");
      // TODO(stevescott): is there a way to find the name of
      // offending argument in R, so that I can provide a better error
      // message?
    }
    SEXP dims = PROTECT(Rf_getAttrib(matrix, R_DimSymbol));
    if(Rf_length(dims) != 2){
      report_error("Wrong number of dimensions in GetMatrixDimensions");
    }
    int *rdims = INTEGER(dims);
    std::pair<int,int> ans = std::make_pair(rdims[0], rdims[1]);
    UNPROTECT(1);
    return ans;
  }

  std::vector<int> GetArrayDimensions(SEXP array) {
    if (!Rf_isArray(array)) {
      report_error("GetArrayDimensions called on a non-array object.");
    }
    SEXP r_dims = PROTECT(Rf_getAttrib(array, R_DimSymbol));
    std::vector<int> dims(Rf_length(r_dims));
    int *rdims = INTEGER(r_dims);
    for (int i = 0; i < dims.size(); ++i) {
      dims[i] = rdims[i];
    }
    UNPROTECT(1);
    return dims;
  }

  std::string GetStringFromList(SEXP my_list, const std::string &name){
    SEXP elt = getListElement(my_list, name);
    if(!Rf_isString(elt)){
      std::ostringstream err;
      err << "There is no string named " << name
          << " in the supplied list." << std::endl;
      report_error(err.str().c_str());
    }
    return CHAR(STRING_ELT(elt, 0));
  }

  ConstVectorView ToBoomVectorView(SEXP v) {
    if (!Rf_isNumeric(v)) {
      report_error("ToBoomVectorView called with a non-numeric argument.");
    }
    PROTECT(v = Rf_coerceVector(v, REALSXP));
    int n = Rf_length(v);
    double *data = REAL(v);
    UNPROTECT(1);
    return ConstVectorView(data, n, 1);
  }

  Vector ToBoomVector(SEXP v){
    return Vector(ToBoomVectorView(v));
  }

  ConstSubMatrix ToBoomMatrixView(SEXP m) {
    if (!Rf_isMatrix(m)) {
      report_error("ToBoomMatrix called with a non-matrix argument");
    }
    std::pair<int,int> dims = GetMatrixDimensions(m);
    PROTECT(m = Rf_coerceVector(m, REALSXP));
    ConstSubMatrix ans(REAL(m), dims.first, dims.second);
    UNPROTECT(1);
    return ans;
  }

  Matrix ToBoomMatrix(SEXP m){
    return Matrix(ToBoomMatrixView(m));
  }

  SpdMatrix ToBoomSpdMatrix(SEXP m){
    return SpdMatrix(ToBoomMatrixView(m));
  }

  DataTable ToBoomDataTable(SEXP r_data_frame) {
    if (!Rf_isFrame(r_data_frame)) {
      report_error("r_data_frame must be a data.frame");
    }
    DataTable table;
    std::vector<std::string> variable_names = getListNames(r_data_frame);
    int number_of_variables = Rf_length(r_data_frame);
    for (int i = 0; i < number_of_variables; ++i) {
      SEXP r_variable = VECTOR_ELT(r_data_frame, i);
      if (Rf_isFactor(r_variable)) {
        Factor factor(r_variable);
        table.append_variable(factor.vector_of_observations(),
                              variable_names[i]);
      } else if (Rf_isString(r_variable)) {
        table.append_variable(make_catdat_ptrs(StringVector(r_variable)),
                              variable_names[i]);
      } else if (Rf_isNumeric(r_variable)) {
        table.append_variable(ToBoomVector(r_variable),
                              variable_names[i]);
      } else {
        std::ostringstream err;
        err << "Variable " << i + 1
            << " in the data frame ("
            << variable_names[i]
            << ") is neither numeric, factor, nor character.  "
            << "I'm not sure what to do with it.";
        report_error(err.str());
      }
    }
    return table;
  }

  std::vector<bool> ToVectorBool(SEXP logical_vector){
    if(!Rf_isVector(logical_vector)) {
      report_error("ToVectorBool requires a logical vector argument.");
    }
    PROTECT(logical_vector = Rf_coerceVector(logical_vector, LGLSXP));
    int n = Rf_length(logical_vector);
    std::vector<bool> ans(n);
    int *data = LOGICAL(logical_vector);
    ans.assign(data, data + n);
    UNPROTECT(1);
    return ans;
  }

  std::vector<int> ToIntVector(SEXP r_int_vector) {
    if (!Rf_isInteger(r_int_vector)) {
      report_error("Argument to ToIntVector must be a vector of integers.");
    }
    int *values = INTEGER(r_int_vector);
    int length = Rf_length(r_int_vector);
    return std::vector<int>(values, values + length);
  }

  SEXP ToRVector(const Vector &v){
    int n = v.size();
    SEXP ans;
    PROTECT(ans = Rf_allocVector(REALSXP, n));
    double *data = REAL(ans);
    for(int i = 0; i < n; ++i) data[i] = v[i];
    UNPROTECT(1);
    return ans;
  }

  SEXP ToRMatrix(const Matrix &m){
    SEXP ans;
    PROTECT(ans = Rf_allocMatrix(REALSXP, m.nrow(), m.ncol()));
    double *data = REAL(ans);
    std::copy(m.begin(), m.end(), data);
    UNPROTECT(1);
    return ans;
  }

  SEXP ToRMatrix(const Matrix &m,
                 const std::vector<std::string> &rownames,
                 const std::vector<std::string> &colnames){
    if (!rownames.empty() && rownames.size() != m.nrow()) {
      report_error("In ToRMatrix:  Vector of row names does not match "
                   "the number of rows in m.");
    } else if (!colnames.empty() && colnames.size() != m.ncol()) {
      report_error("In ToRMatrix:  Vector of column names does not match "
                   "the number of columns in m.");
    }
    SEXP ans;
    PROTECT(ans = Rf_allocMatrix(REALSXP, m.nrow(), m.ncol()));
    double *data = REAL(ans);
    std::copy(m.begin(), m.end(), data);

    SEXP r_dimnames;
    PROTECT(r_dimnames = Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(
        r_dimnames,
        0,
        rownames.empty() ? R_NilValue : CharacterVector(rownames));
    SET_VECTOR_ELT(
        r_dimnames,
        1,
        colnames.empty() ? R_NilValue : CharacterVector(colnames));
    Rf_dimnamesgets(ans, r_dimnames);
    UNPROTECT(2);
    return ans;
  }

  SEXP ToRMatrix(const LabeledMatrix &m) {
    return ToRMatrix(m, m.row_names(), m.col_names());
  }

  std::string ToString(SEXP r_string) {
    if (TYPEOF(r_string) == CHARSXP) {
      return CHAR(r_string);
    } else if(Rf_isString(r_string)){
      return CHAR(STRING_ELT(r_string, 0));
    } else {
      report_error("ToString could not convert its argument to a string");
    }
    return "";
  }

  Factor::Factor(SEXP r_factor)
      : values_(Rf_length(r_factor)),
        levels_(new CatKey(GetFactorLevels(r_factor)))
  {
    if (Rf_isFactor(r_factor)) {
      int * factor_numeric_values = INTEGER(r_factor);
      for (int i = 0; i < values_.size(); ++i) {
        values_[i] = factor_numeric_values[i] - 1;
      }
    } else {
      report_error("A C++ Factor can only be created from an R factor.");
    }
  }

  int Factor::length() const {
    return values_.size();
  }

  int Factor::number_of_levels() const {
    return levels_->size();
  }

  int Factor::operator[](int i) const {
    return values_[i];
  }

  CategoricalData Factor::to_cateogrical_data(int i) const {
    return CategoricalData(values_[i], levels_);
  }

  std::vector<Ptr<CategoricalData> > Factor::vector_of_observations() const {
    std::vector<Ptr<CategoricalData> > ans;
    ans.reserve(this->length());
    for (int i = 0; i < length(); ++i) {
      ans.push_back(new CategoricalData(values_[i], levels_));
    }
    return ans;
  }

  //======================================================================
  RErrorReporter::~RErrorReporter() {
    if (error_message_) {
      // Build the error message in memory managed by R, which will be
      // freed when Rf_error is called.
      SEXP s_error_message = PROTECT(Rf_mkChar(error_message_->c_str()));
      // Then free the memory we're holding for the error message.
      delete error_message_;
      Rf_error("%s", CHAR(s_error_message));
    }
  }

  void RErrorReporter::SetError(const std::string &error) {
    if (!error_message_) {
      // If there are multple error messages, only the first is kept.
      error_message_ = new std::string(error);
    }
  }

  namespace {
    // Wrapper for R_CheckUserInterrupt.
    static void check_interrupt_func(void *dummy) {
      R_CheckUserInterrupt();
    }
  }  // namespace

  bool RCheckInterrupt() {
    // Checking in this way will ensure that R_CheckUserInterrupt will not
    // longjmp out of the current context, so C++ can clean up correctly.
    return (!R_ToplevelExec(check_interrupt_func, NULL));
  }

}  // namespace BOOM;
