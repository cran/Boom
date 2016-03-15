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

#include <Models/PosteriorSamplers/MvnConjSampler.hpp>
#include <Models/MvnModel.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef MvnConjSampler MCS;

  MCS::MvnConjSampler(MvnModel *Mod, const Vector &mu0,
              double kappa, const SpdMatrix & SigmaHat,
              double prior_df,
          RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mod_(Mod),
      mu_(new MvnGivenSigma(mu0, kappa, mod_->Sigma_prm() )),
      siginv_(new WishartModel(prior_df, SigmaHat))
  {
  }

  MCS::MvnConjSampler(MvnModel *Mod,
              Ptr<MvnGivenSigma> Mu,
              Ptr<WishartModel> Siginv,
          RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mod_(Mod),
      mu_(Mu),
      siginv_(Siginv)
  {
    mu_->set_Sigma(mod_->Sigma_prm());
  }

  double MCS::logpri()const{
    double ans = siginv_->logp(mod_->siginv());
    ans += mu_->logp(mod_->mu());
    return ans;
  }

  const Vector & MCS::mu0()const{ return mu_->mu();}
  double MCS::kappa()const{ return mu_-> kappa();}
  double MCS::prior_df()const{ return siginv_->nu();}
  const SpdMatrix & MCS::prior_SS()const{ return siginv_->sumsq();}

  void MCS::set_posterior_sufficient_statistics(){
    Ptr<MvnSuf> s = mod_->suf();
    n = s->n();
    k = kappa();
    const Vector & mu0(this->mu0());

    Vector ybar = s->ybar();
    mu_hat = ybar;
    mu_hat *= (n/k);
    mu_hat += mu0;
    mu_hat *= k/(n+k);

    SS = prior_SS();
    SS += s->center_sumsq();
    SS.add_outer( ybar-mu_hat, n);
    SS.add_outer( mu0-mu_hat, k);

    DF = prior_df() + n;
  }

  void MCS::draw(){
    set_posterior_sufficient_statistics();
    SS = rWish(DF, SS.inv());// check this.. inverse?
    mod_->set_siginv(SS);
    mu_hat = rmvn_mt(rng(), mu_hat, mod_->Sigma()/(n+k));
    mod_->set_mu(mu_hat);
  }

  void MCS::find_posterior_mode(double){
    set_posterior_sufficient_statistics();
    mod_->set_mu(mu_hat);
    double scale_factor = (DF - SS.nrow()-1);
    if(scale_factor<0) scale_factor=0;
    SS *= scale_factor;
    mod_->set_siginv(SS);
  }

}  // namespace BOOM
