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
#ifndef BOOM_TRUNCATED_GAMMA_MODEL_HPP
#define BOOM_TRUNCATED_GAMMA_MODEL_HPP

#include <Models/GammaModel.hpp>

namespace BOOM{

// this is not a fully fledged model, because we don't have inference
// worked out for it yet.  Loglike depends on nc_

class TruncatedGammaModel
    : public GammaModel
{
 public:
  TruncatedGammaModel(double a, double b, double trunc);
  double logp(double x)const override;

 private:
  double trunc_;
  double lognc_;
};

}
#endif// BOOM_TRUNCATED_GAMMA_MODEL_HPP
