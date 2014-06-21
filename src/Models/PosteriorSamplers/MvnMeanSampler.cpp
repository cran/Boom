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
#include <Models/PosteriorSamplers/MvnMeanSampler.hpp>
#include <Models/MvnModel.hpp>
#include <Models/ParamTypes.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
namespace BOOM{

  typedef MvnConjMeanSampler MCS;

  MCS::MvnConjMeanSampler(MvnModel *Mod)
    : mvn(Mod),
      mu0(new VectorParams(Mod->mu().zero())),
      kappa(new UnivParams(0.0))
  {}

  MCS::MvnConjMeanSampler
  (MvnModel *Mod, Ptr<VectorParams> Mu0, Ptr<UnivParams> Kappa)
    : mvn(Mod),
      mu0(Mu0),
      kappa(Kappa)
  {}

  MCS::MvnConjMeanSampler
  (MvnModel *Mod, const Vec & Mu0, double Kappa)
    : mvn(Mod),
      mu0(new VectorParams(Mu0)),
      kappa(new UnivParams(Kappa))
  {}

  void MCS::draw(){
    Ptr<MvnSuf> s = mvn->suf();
    double n =s->n();
    double k = kappa->value();
    const Spd & Siginv(mvn->siginv());
    Spd ivar = (n+k)*Siginv;
    double w = n/(n+k);
    Vec mu = w*s->ybar() + (1.0-w)*mu0->value();
    mu = rmvn_ivar_mt(rng(), mu, ivar);
    mvn->set_mu(mu);
  }

  double MCS::logpri()const{
    double k = kappa->value();
    if(k==0.0) return BOOM::negative_infinity();
    const Ptr<SpdParams> Sig = mvn->Sigma_prm();
    const Vec &mu(mvn->mu());
    uint d = mvn->dim();
    double ldsi = d*log(k) + Sig->ldsi();
    return dmvn(mu, mu0->value(), k*Sig->ivar(), ldsi, true);
  }

  //----------------------------------------------------------------------
  typedef MvnMeanSampler MMS;

  MMS::MvnMeanSampler(MvnModel *m, Ptr<VectorParams> Mu0, Ptr<SpdParams> Omega)
    : mvn(m),
      mu_prior_(new MvnModel(Mu0, Omega))
  {}

  MMS::MvnMeanSampler(MvnModel *m, Ptr<MvnBase> Pri)
    : mvn(m),
      mu_prior_(Pri)
  {}

  MMS::MvnMeanSampler(MvnModel *m, const Vec &Mu0, const Spd &Omega)
    : mvn(m),
      mu_prior_(new MvnModel(Mu0, Omega))
  {}

  double MMS::logpri()const{
    return mu_prior_->logp(mvn->mu());
  }

  void MMS::draw(){
    Ptr<MvnSuf> s = mvn->suf();
    double n = s->n();
    const Spd &siginv(mvn->siginv());
    const Spd &ominv(mu_prior_->siginv());
    Spd Ivar = n*siginv + ominv;
    Vec mu = Ivar.solve(n*(siginv*s->ybar()) + ominv*mu_prior_->mu());
    mu = rmvn_ivar(mu, Ivar);
    mvn->set_mu(mu);
  }
}
