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

// #include <loki/static_check.h>


namespace BOOM{
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
