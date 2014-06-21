/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include <Models/TruncatedGammaModel.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

typedef TruncatedGammaModel TGM;

TGM::TruncatedGammaModel(double a, double b, double trunc)
    : GammaModel(a,b),
      trunc_(trunc),
      lognc_(pgamma(trunc_, a, b, false, true))
{}

double TGM::logp(double x)const{
  if(x < trunc_) return BOOM::negative_infinity();
  return dgamma(x,alpha(),beta(),true) - lognc_;
}
}
