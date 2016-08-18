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
#include <distributions.hpp>

namespace BOOM{

  typedef ZeroMeanGaussianConjSampler ZGS;

  ZGS::ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel * mod,
                                   Ptr<GammaModelBase> ivar,
                                   RNG &seeding_rng)
      : GaussianVarSampler(mod, ivar, seeding_rng),
        model_(mod)
  {}

  ZGS::ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel * mod,
                                   double df, double sigma_guess,
                                   RNG &seeding_rng)
      : GaussianVarSampler(mod, new ChisqModel(df, sigma_guess)),
        model_(mod)
  {}

  ZGS * ZGS::clone()const{ return new ZGS(*this);}

  // The logic of the posterior mode here is as follows.  The prior is
  // a Gamma on 1/sigsq, but the parameter we really care about is
  // sigsq.  This introduces a Jacobian term that needs to be taken
  // account of in the optimization.  The mode of the gamma
  // distribution is (a-1)/b, but the mode of the inverse gamma
  // distribution is b/(a+1).
  //
  // The deciding factor is that the prior on sigsq is a Gamma model
  // on 1/sigsq and not an inverse Gamma model on sigsq.  For this
  // result to agree with numerical optimizers we need to do the
  // optimization with respect to 1/sigsq.
  void ZGS::find_posterior_mode(double){
    double a = ivar()->alpha() + .5 * model_->suf()->n();
    double b = ivar()->beta() + .5 * model_->suf()->sumsq();
    model_->set_sigsq(b/(a+1));   // with respect to 1.0 / sigsq
  }


  //
  double ZGS::log_posterior(double sigsq, double &d1, double &d2,
                            uint nd) const {
    // The log likelihood is already parameterized with respect to
    // sigma^2, so derivatives are easy.
    double logp = model_->log_likelihood(sigsq,
                                         nd > 0 ? &d1 : nullptr,
                                         nd > 1 ? &d2 : nullptr);

    double a = ivar()->alpha();
    double b = ivar()->beta();
    // The log prior is the gamma density plus a jacobian term:
    // log(abs(d(siginv) / d(sigsq))).
    logp += dgamma(1/sigsq, a, b, true) - 2 * log(sigsq);
    if (nd > 0) {
      double sig4 = sigsq * sigsq;
      d1 += -(a + 1) / sigsq + b / sig4;
      if (nd > 1) {
        d2 += (a + 1) / sig4 - 2 * b / (sig4 * sigsq);
      }
    }
    return logp;
  }

}  // namespace BOOM
