/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_MIXTURE_SAMPLER_TIM_HPP_
#define BOOM_BINOMIAL_MIXTURE_SAMPLER_TIM_HPP_

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/BinomialLogitModel.hpp>
#include <Models/MvnBase.hpp>

#include <Samplers/TIM.hpp>

namespace BOOM{

  class BinomialLogitSamplerTim : public PosteriorSampler{
   public:
    BinomialLogitSamplerTim(BinomialLogitModel *model,
                            Ptr<MvnBase> prior,
                            bool mode_is_stable = true,
                            double nu = 3);

    virtual void draw();
    virtual double logpri()const;

    double logp(const Vec &beta)const;
    double dlogp(const Vec &beta, Vec &g)const;
    double d2logp(const Vec &beta, Vec &g, Mat &H)const;
    double Logp(const Vec &beta, Vec &g, Mat &h, int nd)const;
   private:
    BinomialLogitModel *m_;
    Ptr<MvnBase> pri_;
    TIM sam_;
    //    Ptr<MvtIndepProposal> create_proposal(int n)const;
  };

}

#endif //  BOOM_BINOMIAL_MIXTURE_SAMPLER_TIM_HPP_
