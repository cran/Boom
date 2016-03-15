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

#ifndef BOOM_MVN_CONJ_SAMPLER_HPP
#define BOOM_MVN_CONJ_SAMPLER_HPP

#include <Models/MvnGivenSigma.hpp>
#include <Models/WishartModel.hpp>

namespace BOOM{

  class MvnModel;

  class MvnConjSampler
    : public PosteriorSampler
  {
  public:
    MvnConjSampler(MvnModel *mod, const Vector &mu0, double kappa,
           const SpdMatrix & SigmaHat, double prior_df,
       RNG &seeding_rng = GlobalRng::rng);

    MvnConjSampler(MvnModel *mod, Ptr<MvnGivenSigma>, Ptr<WishartModel>,
       RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri()const override;
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override {
      return true;
    }
    double kappa()const;
    double prior_df()const;
    const Vector & mu0()const;
    const SpdMatrix & prior_SS()const;
  private:
    MvnModel *mod_;
    Ptr<MvnGivenSigma> mu_;
    Ptr<WishartModel> siginv_;

    mutable SpdMatrix SS;
    mutable Vector mu_hat;
    mutable double n,k,DF;

    void set_posterior_sufficient_statistics();
  };

}  // namespace BOOM
#endif// BOOM_MVN_CONJ_SAMPLER_HPP
