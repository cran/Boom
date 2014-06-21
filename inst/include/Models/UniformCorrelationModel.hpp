/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_UNIFORM_CORRELATION_MODEL_HPP
#define BOOM_UNIFORM_CORRELATION_MODEL_HPP
#include <Models/Policies/NullParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/SpdParams.hpp>

namespace BOOM{

  class UniformCorrelationModel
    : public NullParamPolicy,
      public IID_DataPolicy<SpdParams>,
      public PriorPolicy,
      public CorrModel
  {
  public:
    UniformCorrelationModel(uint dim);
    UniformCorrelationModel(const UniformCorrelationModel &);
    UniformCorrelationModel * clone()const;

    void initialize_params();
    double pdf(Ptr<Data>, bool logscale)const;
    double logp(const Corr &)const;

    uint dim()const;
    Corr sim()const;
  private:
    uint dim_;
  };
}
#endif// BOOM_UNIFORM_CORRELATION_MODEL_HPP
