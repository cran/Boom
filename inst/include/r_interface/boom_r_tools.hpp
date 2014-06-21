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

#ifndef BOOM_R_TOOLS_HPP_
#define BOOM_R_TOOLS_HPP_

#include <string>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>

#include <Models/CategoricalData.hpp>

//======================================================================
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

// If Rinternals.h has already been included and R_NO_REMAP has not
// yet been defined, then we throw a compiler error to prevent madness
// caused by R's preprocessor renaming of things like length() and
// error().
#ifndef R_NO_REMAP
#define R_NO_REMAP
#ifdef R_INTERNALS_H_
#error Code that includes both boom_r_tools.hpp and Rinternals.h must either\
 (a) include them in that order or (b) define R_NO_REMAP.
#endif  // ifdef R_INTERNALS_H_
#endif  // ifndef R_NO_REMAP

#include <Rinternals.h>

namespace BOOM{
  // Returns list[[name]] if a list element with that name exists.
  // Returns R_NilValue otherwise.
  SEXP getListElement(SEXP list, const std::string &name);

  // Extract the names from a list.  If the list has no names
  // attribute a vector of empty strings is returned.
  std::vector<std::string> getListNames(SEXP list);

  // Set the names attribute of 'list' to match 'list_names'.  BOOM's
  // error reporting mechanism is invoked if the length of 'list' does
  // not match the length of 'list_names'.
  // Returns 'list' with the new and improved set of 'list_names'.
  SEXP setListNames(SEXP list, const std::vector<std::string> &list_names);

  // Returns the levels attribute of the factor argument.  Throws an
  // exception if the argument is not a factor.
  std::vector<std::string> GetFactorLevels(SEXP factor);

  // Converts an R character vector into a c++ vector of strings.  If
  // the object is NULL an empty string vector is returned.  Otherwise
  // if the object is not a character vector an exception is thrown.
  std::vector<std::string> StringVector(SEXP r_character_vector);

  // Converts a C++ vector of strings into an R character vector.
  SEXP CharacterVector(const std::vector<std::string> &string_vector);

  // Creates a new list with the contents of the original 'list' with
  // new_element added.  The names of the original list are copied,
  // and 'name' is appended.  The original 'list' is not modified, so
  // it is possible to write:
  // my_list = appendListElement(my_list, new_thing, "blah");
  // Two things to note:
  // (1) The output is in new memory, so it is not PROTECTED by default
  // (2) Each time you call this function all the list entries are
  //     copied, and (more importantly) new space is allocated, so
  //     you're better off creating the list you want from the
  //     beginning if you can.
  SEXP appendListElement(SEXP list, SEXP new_element, const std::string &name);

  // Appends the collection of SEXP elements to the list.
  // Args:
  //   list:  The original list.
  //   new_elements:  The vector of new elements to add to the list.
  //   new_element_names: A vector of names for the new elements.  The
  //     length of this vector MUST match the length of the new
  //     elements.
  //
  // Returns:
  //   A new list containing the elements from the original list, with
  //   new_elements appended at the end.
  //   *** NOTE ***
  //   The return value is in new memory, so even if the name of the
  //   list is the same, it will have to be re-PROTECTed.
  SEXP appendListElements(
      SEXP list,
      const std::vector<SEXP> &new_elements,
      const std::vector<std::string> &new_element_names);

  // Creates a list from the C++ vector of SEXP elements.  The vector
  // of element_names can be empty, in which case the list will not
  // have names.  If non-empty, then the length of element_names must
  // match the length of elements.
  SEXP CreateList(
      const std::vector<SEXP> &elements,
      const std::vector<std::string> &element_names);

  // Returns the class attribute of the specified R object.  If no
  // class attribute exists an empty vector is returned.
  std::vector<std::string> GetS3Class(SEXP object);

  // Returns a pair, with .first set to the number of rows, and
  // .second set to the number of columns.  If the argument is not a
  // matrix then an exception will be thrown.
  std::pair<int, int> GetMatrixDimensions(SEXP matrix);

  // Returns a vector of dimensions for an R multi-way array.  If the
  // argument is not an array, then an exception will be thrown.
  std::vector<int> GetArrayDimensions(SEXP array);

  // If 'my_list' contains a character vector named 'name' then the
  // first element of that character vector is returned.  If not then
  // an exception will be thrown.
  std::string GetStringFromList(SEXP my_list, const std::string &name);

  // If 'my_vector' is a numeric vector, it is converted to a BOOM::Vector.
  // Otherwise an exception will be thrown.
  Vector ToBoomVector(SEXP my_vector);

  // If 'r_matrix' is an R matrix, it is converted to a BOOM::Matrix.
  // Otherwise an exception will be thrown.
  Matrix ToBoomMatrix(SEXP r_matrix);

  // If 'my_matrix' is an R matrix, it is converted to a BOOM::Spd.  If
  // the conversion fails then an exception will be thrown.
  Spd ToBoomSpd(SEXP my_matrix);

  // If 'my_vector' is an R logical vector, then it is converted to a
  // std::vector<bool>.  Otherwise an exception will be thrown.
  std::vector<bool> ToVectorBool(SEXP my_vector);

  // Convert a BOOM vector or matrix to its R equivalent.  Less type
  // checking is needed for these functions than in the other
  // direction because we know the type of the input.
  SEXP ToRVector(const Vector &boom_vector);
  SEXP ToRMatrix(const Matrix &boom_matrix);

  // This version produces an R matrix with row names and column
  // names.  A zero-length vector indicates that no names are desired
  // for that dimension.  Otherwise the size of row_names must equal
  // the number of rows in boom_matrix, and likewise for col_names.
  SEXP ToRMatrix(const Matrix &boom_matrix,
                 const std::vector<std::string> &row_names,
                 const std::vector<std::string> &col_names);
  SEXP ToRMatrix(const LabeledMatrix &boom_labeled_matrix);

  std::string ToString(SEXP r_string);

  // A Factor object is intended to be intialized with an R factor.
  class Factor {
   public:
    Factor(SEXP r_factor);

    // Corresponds to R's length(r_factor).
    int length() const;

    // Corresponds to R's length(levels(r_factor))
    int number_of_levels() const;

    // Returns the integer value of observation i.  Note that in R,
    // observation i is 1-based, while here it is zero-basd.
    int operator[](int i) const;

    // Returns a BOOM::CategoricalData corresponding to observation i.
    CategoricalData to_cateogrical_data(int i)const;

   private:
    std::vector<int> values_;
    Ptr<CatKey> levels_;
  };

}

#endif  // BOOM_R_TOOLS_HPP_
