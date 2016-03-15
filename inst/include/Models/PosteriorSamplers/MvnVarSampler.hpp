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
#ifndef BOOM_MVN_VAR_SAMPLER_HPP
#define BOOM_MVN_VAR_SAMPLER_HPP
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/SpdParams.hpp>

namespace BOOM{
  class MvnModel;
  class WishartModel;

  class MvnVarSampler : public PosteriorSampler{
    // assumes y~N(mu, Sigma), with Sigma^-1~W(df, SS).  The prior on
    // mu may or may not be conjugate.  The sampling step will
    // condition on mu.  Use MvnConjVarSampler if you want to
    // integrate out mu.
  public:
    MvnVarSampler(MvnModel *, double df, const SpdMatrix & SS,
                  RNG &seeding_rng = GlobalRng::rng);
    MvnVarSampler(MvnModel *, const WishartModel &siginv_prior,
                  RNG &seeding_rng = GlobalRng::rng);
    MvnVarSampler(MvnModel *, RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;
  private:
    MvnModel *mvn_;
    Ptr<UnivParams> pdf_;
    Ptr<SpdParams> pss_;
   protected:
    MvnModel * mvn(){return mvn_;}
    const SpdParams * pss()const{return pss_.get();}
    const UnivParams * pdf()const{return pdf_.get();}
  };

  class MvnConjVarSampler : public MvnVarSampler{
    // assumes y~N(mu, Sigma), with mu|Sigma \norm(mu0, Sigma/kappa)
    // and Sigma^-1~W(df, SS)
  public:
    MvnConjVarSampler(MvnModel *, double df, const SpdMatrix & SS,
                      RNG &seeding_rng = GlobalRng::rng);
    MvnConjVarSampler(MvnModel *, const WishartModel &siginv_prior,
                      RNG &seeding_rng = GlobalRng::rng);
    MvnConjVarSampler(MvnModel *, RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
  };

}
#endif// BOOM_MVN_VAR_SAMPLER_HPP
