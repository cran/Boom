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

#include <Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>
#include <Models/GammaModel.hpp>
#include <Models/ChisqModel.hpp>

namespace BOOM{

  typedef ZeroMeanGaussianConjSampler ZGS;

  ZGS::ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel * mod,
                                   Ptr<GammaModelBase> ivar,
                                   RNG &seeding_rng)
      : GaussianVarSampler(mod, ivar, seeding_rng),
        mod(mod)
  {}

  ZGS::ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel * mod,
                                   double df, double sigma_guess,
                                   RNG &seeding_rng)
      : GaussianVarSampler(mod, new ChisqModel(df, sigma_guess)),
        mod(mod)
  {}

  ZGS * ZGS::clone()const{ return new ZGS(*this);}

  void ZGS::find_posterior_mode(double){
    double a = ivar()->alpha() + .5 * mod->suf()->n();
    double b = ivar()->beta() + .5 * mod->suf()->sumsq();
    mod->set_sigsq(b/(1+a));   // with respect to sigsq
  }

}  // namespace BOOM
