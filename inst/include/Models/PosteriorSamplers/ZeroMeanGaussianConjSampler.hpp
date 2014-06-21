/*
  Copyright (C) 2008 Steven L. Scott

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
#ifndef BOOM_ZERO_MEAN_GAUSSIAN_CONJ_SAMPLER_HPP_
#define BOOM_ZERO_MEAN_GAUSSIAN_CONJ_SAMPLER_HPP_

#include <Models/PosteriorSamplers/GaussianVarSampler.hpp>

namespace BOOM{
   class ZeroMeanGaussianModel;

   class ZeroMeanGaussianConjSampler
       : public GaussianVarSampler
   {
    public:
     ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel * mod,
                                 Ptr<GammaModelBase>);
     ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel * mod,
                                 double df, double sigma_guess);

     ZeroMeanGaussianConjSampler * clone()const;
     // The posterior mode is with respect to d sigsq, not d siginv.
     void find_posterior_mode();
    private:
     ZeroMeanGaussianModel * mod;
   };
}
#endif // BOOM_ZERO_MEAN_GAUSSIAN_CONJ_SAMPLER_HPP_
