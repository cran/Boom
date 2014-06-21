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
#ifndef BOOM_CHOL_HPP
#define BOOM_CHOL_HPP

#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>

namespace BOOM{
    class Chol{
    public:
      Chol(const Matrix &A);
      uint nrow()const;
      uint ncol()const;
      uint dim()const;
      Matrix getL()const;
      Matrix getLT()const;
      Matrix solve(const Matrix &B)const;
      Vector solve(const Vector &b)const;
      SpdMatrix inv()const;  // inverse of A
      SpdMatrix original_matrix()const;
      double det()const;     // det(A)
      double logdet()const;  // log(det(A))
      bool is_pos_def()const{return pos_def;}
      Chol & operator *= (double a);
     private:
      Matrix dcmp;
      bool pos_def;
      bool zeros_;  // true if upper diagonal has been zeroed out;
      void check()const;
    };

  Chol operator*(double a, const Chol &C);
  Chol operator*(const Chol &C, double a);

}
#endif// BOOM_CHOL_HPP
