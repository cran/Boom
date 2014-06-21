/*
  Copyright (C) 2005 Steven L. Scott

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
#ifndef BOOM_NEWLA_DIAGONAL_MATRIX_HPP
#define BOOM_NEWLA_DIAGONAL_MATRIX_HPP
#include <vector>
#include <boost/shared_ptr.hpp>
#include <iosfwd>
#include <LinAlg/Matrix.hpp>

namespace BOOM{
    using std::ostream;
    using std::istream;

    class Vector;
    class VectorView;
    class SpdMatrix;
    class DiagonalMatrix : public Matrix{
      Matrix & rbind(const Matrix &){return *this;}
      Matrix & rbind(const Vector &){return *this;}
      Matrix & cbind(const Matrix &){return *this;}
      Matrix & cbind(const Vector &){return *this;}
      void set_row(uint , const Vector &){}
      void set_row(uint , const double *){}
      void set_row(uint , double ){}
      void set_col(uint , const Vector &){}
      void set_col(uint , const double *){}
      void set_col(uint , double ){}
      void set_rc(uint ,  double ){}  // sets row and column i to x
      Matrix & add_outer(const Vector &, const Vector &, double){return *this;}

      Matrix & operator+=(const Matrix &){return *this;}
      Matrix & operator-=(const Matrix &){return *this;}

    public:
      DiagonalMatrix();
      DiagonalMatrix(uint n, double x=0.0);
      template <class VEC>
      explicit DiagonalMatrix(const VEC &v);
      DiagonalMatrix(const DiagonalMatrix &);       // reference semantics
      DiagonalMatrix(const Matrix &);                // value semantics

      DiagonalMatrix & operator=(const DiagonalMatrix &); // value semantics
      DiagonalMatrix & operator=(const Matrix &);                   // value semantics
      DiagonalMatrix & operator=(const double &);                   // value semantics

      bool operator==(const DiagonalMatrix &)const;

      void swap(DiagonalMatrix &rhs); // efficient.. swaps pointers and size info
      virtual void randomize();  // fills entries with U(0,1) random variables.

      //---- change size and shape  -----
      DiagonalMatrix & resize(uint n);

      //-------- subscripting, range checking can be turned off
      //-------- by defining the macro NDEBUG
      double & operator[](uint n);               // returns diagonal element n
      const double & operator[](uint n)const;

      //------ linear algebra...

      virtual Matrix & mult(const Matrix &B, Matrix &ans, double scal=1.0)const;  // this * B
      virtual Matrix & Tmult(const Matrix &B, Matrix &ans, double scal=1.0)const; // this^T * B
      virtual Matrix & multT(const Matrix &B, Matrix &ans, double scal=1.0)const; // this * B^T

      virtual Matrix & mult(const SpdMatrix &S, Matrix & ans, double scal=1.0)const;
      virtual Matrix & Tmult(const SpdMatrix &S, Matrix & ans, double scal=1.0)const;
      virtual Matrix & multT(const SpdMatrix &S, Matrix & ans, double scal=1.0)const;
      // no BLAS support for this^T * S
      // virtual Matrix & Tmult(const SpdMatrix &S, Matrix & ans, double scal=1.0)const;

      virtual DiagonalMatrix & mult(const DiagonalMatrix &B, Matrix &ans, double scal=1.0)const;
      virtual DiagonalMatrix & Tmult(const DiagonalMatrix &B, Matrix &ans, double scal=1.0)const;
      virtual DiagonalMatrix & multT(const DiagonalMatrix &B, Matrix &ans, double scal=1.0)const;

      virtual Vector & mult(const Vector &v, Vector &ans, double scal=1.0)const;   // this * v
      virtual Vector & Tmult(const Vector &v, Vector &ans, double scal=1.0)const;  // this^T * v

      //      Matrix Id() const;
      DiagonalMatrix t() const;
      DiagonalMatrix  inv() const;
      SpdMatrix inner() const;   // returns X^tX

      Matrix solve(const Matrix &mat) const;
      Vector solve(const Vector &v) const;
      double det() const;
      Vector singular_values()const; // sorted largest to smallest
      uint rank(double prop=1e-12) const;
      // 'rank' is the number of singular values at least 'prop' times
      // the largest
      Vector real_evals()const;

      //--------  Math -------------
      DiagonalMatrix & operator+=(double x);
      DiagonalMatrix & operator*=(double x);
      DiagonalMatrix & operator-=(double x);
      DiagonalMatrix & operator/=(double x);

      double sum()const;
      double prod()const;

    };

    template <class VEC>
    DiagonalMatrix::DiagonalMatrix(const VEC &v)
      : Matrix(v.size(), v.size())
    {
      std::copy(v.begin(), v.end(), dbegin());
    }

}
#endif // BOOM_NEWLA_MATRIX_HPP
