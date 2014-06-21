/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_REGRESSION_CONJUGATE_SAMPLER_HPP
#define BOOM_REGRESSION_CONJUGATE_SAMPLER_HPP

#include <Models/Glm/RegressionModel.hpp>
#include <Models/Glm/MvnGivenXandSigma.hpp>
#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM{
  class RegressionConjSampler
    : public PosteriorSampler
  {
    // for drawing p(beta, sigma^2 | y)
    // prior is p(beta | sigma^2, X) = N(b0, sigsq * XTX/kappa)
    //          p(sigsq | X) = Gamma(prior_df/2, prior_ss/2)


  public:
    RegressionConjSampler(RegressionModel *,
                          Ptr<MvnGivenXandSigma>,
			  Ptr<GammaModelBase>);
    virtual void draw();
    virtual double logpri()const;

    void find_posterior_mode();

    const Vec & b0()const;
    double kappa()const;
    double prior_df()const;
    double prior_ss()const;
  private:
    RegressionModel *m_;
    Ptr<MvnGivenXandSigma> mu_;
    Ptr<GammaModelBase> siginv_;
    Vec beta_tilde;
    Spd ivar;
    double SS, DF;
    void set_posterior_suf();
  };
}
#endif// BOOM_REGRESSION_CONJUGATE_SAMPLER_HPP
