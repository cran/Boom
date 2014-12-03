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

#ifndef BOOM_SWEEP_HPP
#define BOOM_SWEEP_HPP

#include <vector>
#include<BOOM.hpp>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>

namespace BOOM{

  // A SweptVarianceMatrix is a matrix that has been operated on by
  // the SWEEP operator.  The SWEEP operator operates on multivariate
  // normal parameters (or sufficient statistics).  If Sigma is the
  // variance matrix of a zero-mean multivariate normal, then
  // SWP[k](Sigma) moves element k from the unobserved, random 'Y'
  // side of the equation, to the observed, conditional, 'X' side.
  //
  // Suppose the matrix A is  A = (A_11  A_12)
  //                              (A_21  A_22)
  //
  // Then sweeping on the _1 elements of A (you only ever sweep on
  // diagonal elements) yields
  //    SWP[1](A) = ( -A_11^{-1}             A_11^{-1} * A_{12}              )
  //                ( A_{21} * A_{11}^{-1}   A_22 - A_{21} A_{11}^{-1} A_{12})
  //

  class SweptVarianceMatrix{
    // sweeping a variable is equivalent to conditioning on it.
    // i.e. when a variable is swept it changes from 'y' to 'x'.
    SpdMatrix S;
    std::vector<bool> swept_;
    uint nswept_;
   public:
    SweptVarianceMatrix();
    SweptVarianceMatrix(uint d);
    SweptVarianceMatrix(const SpdMatrix &m);
    SweptVarianceMatrix(const SweptVarianceMatrix &sm);

    void SWP(uint m);
    void SWP(const std::vector<bool> &);
    void RSW(uint m);

    Matrix Beta()const;  // to compute E(unswept | swept)
    Vector E_y_given_x(const Vector &x, const Vector &mu);
    SpdMatrix V_y_given_x()const;
    SpdMatrix ivar_x()const;

    uint ydim()const;
    uint xdim()const;

    SpdMatrix & swept_matrix(){return S;}
    const SpdMatrix & swept_matrix()const{return S;}
  };
}

#endif // BOOM_SWEEP_HPP
