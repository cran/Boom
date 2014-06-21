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
#ifndef BOOM_CHI_SQUARE_MODEL_HPP
#define BOOM_CHI_SQUARE_MODEL_HPP
#include <Models/GammaModel.hpp>

namespace BOOM{
  class ChisqModel
    : public GammaModelBase,
      public ParamPolicy_2<UnivParams, UnivParams>,
      public PriorPolicy
  {
  public:
    ChisqModel(double df = 1.0, double sigma_est=1.0);
    ChisqModel(const ChisqModel &rhs);
    ChisqModel * clone()const;

    // Sigsq_prm holds expected value
    // Df_prm holds "sample size"

    Ptr<UnivParams> Df_prm();
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Df_prm()const;
    const Ptr<UnivParams> Sigsq_prm()const;

    double df()const;
    double sigsq()const;
    double sum_of_squares()const;

    virtual double alpha()const;
    virtual double beta()const;
    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
  };
}
#endif// BOOM_CHI_SQUARE_MODEL_HPP
