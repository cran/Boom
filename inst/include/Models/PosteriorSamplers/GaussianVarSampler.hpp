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
#ifndef BOOM_GAUSSIAN_VARIANCE_METHOD_HPP
#define BOOM_GAUSSIAN_VARIANCE_METHOD_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp>

namespace BOOM{
  class GaussianModelBase;
  class GammaModelBase;


  // draws sigma given mu

  class GaussianVarSampler : public PosteriorSampler{
   public:
    GaussianVarSampler(GaussianModelBase * m,
                       double prior_df,
                       double prior_sigma_guess);
    GaussianVarSampler(GaussianModelBase * m, Ptr<GammaModelBase> g);
    virtual void draw();
    double logpri()const;
    // Call to ensure that sigma (standard deviation) remains below
    // the specified upper_truncation_point
    void set_sigma_upper_limit(double max_sigma);
   protected:
    const Ptr<GammaModelBase> ivar()const;
   private:
    Ptr<GammaModelBase> gam;
    GaussianModelBase * mod;
    GenericGaussianVarianceSampler sampler_;
  };


}
#endif // BOOM_GAUSSIAN_VARIANCE_METHOD_HPP
