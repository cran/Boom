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

#include <Models/MvtModel.hpp>
#include <Models/WeightedData.hpp>
#include <Models/WeightedMvnModel.hpp>
#include <Models/ScaledChisqModel.hpp>
#include <distributions.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <TargetFun/TargetFun.hpp>
#include <numopt.hpp>
#include <cmath>

namespace BOOM{

  //======================================================================
  typedef MvtModel MVT;

  MVT::MvtModel(uint p, double mu, double sig, double nu)
    : ParamPolicy(),
      DataPolicy(),
      PriorPolicy(),
      mvn(new WeightedMvnModel(p,mu,sig)),
      wgt(new ScaledChisqModel(nu))
  {
    ParamPolicy::add_model(mvn);
    ParamPolicy::add_model(wgt);
  }

  MVT::MvtModel(const Vec &mean, const Spd &Var, double Nu)
    : ParamPolicy(),
      DataPolicy(),
      PriorPolicy(),
      mvn(new WeightedMvnModel(mean, Var)),
      wgt(new ScaledChisqModel(Nu))
  {
    ParamPolicy::add_model(mvn);
    ParamPolicy::add_model(wgt);
  }

  MVT::MvtModel(const MvtModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      VectorModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      LatentVariableModel(rhs),
      LoglikeModel(rhs),
      LocationScaleVectorModel(rhs),
      mvn(rhs.mvn->clone()),
      wgt(rhs.wgt->clone())
  {
    ParamPolicy::add_model(mvn);
    ParamPolicy::add_model(wgt);
  }

  MvtModel *MVT::clone()const{ return new MvtModel(*this);}

  void MVT::initialize_params(){
    mle();
//     double n = suf()->n();
//     Spd Sigma = suf()->sumsq() / n;
//     Vec Mu = suf()->sum()/n;
//     double nu = 30;
//     set_mu(Mu);
//     set_Sigma(Sigma);
//     set_nu(nu);
  }

  Ptr<VectorParams> MVT::Mu_prm(){return mvn->Mu_prm();}
  Ptr<SpdParams> MVT::Sigma_prm(){return mvn->Sigma_prm();}
  Ptr<UnivParams> MVT::Nu_prm(){return wgt->Nu_prm();}

  const Ptr<VectorParams> MVT::Mu_prm()const{
    return mvn->Mu_prm();}
  const Ptr<SpdParams> MVT::Sigma_prm()const{
    return mvn->Sigma_prm();}
  const Ptr<UnivParams> MVT::Nu_prm()const{
    return wgt->Nu_prm();}


  const Vec & MVT::mu()const{ return Mu_prm()->value();}
  const Spd & MVT::Sigma()const{return Sigma_prm()->var();}
  const Spd & MVT::siginv()const{return Sigma_prm()->ivar();}
  double MVT::ldsi()const{ return Sigma_prm()->ldsi();}
  double MVT::nu()const{return Nu_prm()->value();}

  void MVT::set_mu(const Vec &mu){Mu_prm()->set(mu);}
  void MVT::set_Sigma(const Spd &Sig){Sigma_prm()->set_var(Sig);}
  void MVT::set_siginv(const Spd &ivar){Sigma_prm()->set_ivar(ivar);}
  void MVT::set_S_Rchol(const Vec &S, const Mat & L){
    Sigma_prm()->set_S_Rchol(S,L); }
  void MVT::set_nu(double nu){Nu_prm()->set(nu);}

  double MVT::pdf(Ptr<VectorData> dp, bool logscale)const{
    return pdf(dp->value(),logscale);}
  double MVT::pdf(Ptr<Data> dp, bool logscale) const {
    const Vec &v(DAT(dp)->value());
    return pdf(v, logscale);}
  double MVT::pdf(const Vec &x, bool logscale) const{
    return dmvt(x, mu(), siginv(), nu(), ldsi(), logscale); }

  double MVT::logp(const Vec &x)const{return pdf(x, true);}

  void MVT::add_data(Ptr<VectorData> dp){
    DataPolicy::add_data(dp);
    NEW(DoubleData, w)(1.0);
    NEW(WeightedVectorData, v)(dp, w);
    wgt->add_data(w);
    mvn->add_data(v);
  }

  void MVT::add_data(Ptr<Data> dp){
    Ptr<VectorData> d = DAT(dp);
    add_data(d);
  }

  //======================================================================

  double MVT::loglike()const{
    const DatasetType & dat(this->dat());

    double ldsi = this->ldsi();
    const Spd & Siginv(siginv());
    double nu = this->nu();
    double lognu = log(nu);
    const Vec &mu(this->mu());

    const double logpi= 1.1447298858494;
    uint n = dat.size();
    uint d = mu.size();
    double half_npd = .5*(nu+d);

    double ans = lgamma(half_npd)-lgamma(nu/2) - .5*d*(lognu+logpi);
    ans += .5*ldsi + half_npd*lognu;
    ans*=n;

    for(uint i=0; i<n; ++i){
      double delta = Siginv.Mdist(mu, dat[i]->value());
      ans -= half_npd*log(nu + delta/nu);
    }

    return ans;
  }
  //======================================================================

  typedef WeightedVectorData WVD;

  void MVT::Impute(bool sample, RNG &rng){
    std::vector<Ptr<WVD> > &V(mvn->dat());

    for(uint i=0; i<V.size(); ++i){
      Ptr<WVD> d = V[i];
      const Vec &y(d->value());
      double delta = siginv().Mdist(y, mu());
      double a = (nu() + y.length())/2.0;
      double b = (nu() + delta)/2.0;
      double w = sample ? rgamma_mt(rng, a, b) : a/b;
      d->set_weight(w);
    }
    mvn->refresh_suf();
    wgt->refresh_suf();
  }
  void MVT::impute_latent_data(RNG &rng){ Impute(true, rng); }
  void MVT::Estep(){Impute(false, GlobalRng::rng);}

  //------------------------------------------------------------

  class MvtNuTF{
  public:
    MvtNuTF(MvtModel *Mod) : mod(Mod){}
    MvtNuTF * clone()const{return new MvtNuTF(*this);}
    double operator()(const Vec &Nu)const;
    double operator()(const Vec &Nu, Vec &g)const;
  private:
    double Loglike(const Vec &Nu, Vec &g, uint nd)const;
    MvtModel *mod;
  };

  double MvtNuTF::operator()(const Vec &Nu)const{
    Vec g;
    return Loglike(Nu, g, 0);}
  double MvtNuTF::operator()(const Vec &Nu, Vec &g)const{
    return Loglike(Nu,g,1);}

  double MvtNuTF::Loglike(const Vec &Nu, Vec &g, uint nd)const{

    const std::vector<Ptr<VectorData> > & dat(mod->dat());

    double ldsi = mod->ldsi();
    const Spd & Siginv(mod->siginv());
    const Vec &mu(mod->mu());
    const double logpi= 1.1447298858494;
    double nu = Nu[0];
    double lognu = log(nu);
    uint n = dat.size();
    uint d = mu.size();
    double half_npd = .5*(nu+d);

    double ans = lgamma(half_npd)-lgamma(nu/2) - .5*d*(lognu+logpi);
    ans += .5*ldsi + half_npd*lognu;
    ans*=n;

    if(nd>0){
      g[0] = .5*(digamma(half_npd) - digamma(nu/2.0) - d/nu);
      g[0] +=  half_npd/nu+ .5*lognu;
      g[0] *= n;
    }

    for(uint i=0; i<n; ++i){
      double delta = Siginv.Mdist(mu, dat[i]->value());
      double npd = nu+delta;
      ans -= half_npd*log(npd);
      if(nd>0){
	g[0] -=  half_npd/npd + .5*log(npd);
      }
    }
    return ans;
  }
  //------------------------------------------------------------

  void MVT::mle(){
    const double eps = 1e-5;
    double dloglike= eps+1;
    double loglike = this->loglike();
    double old = loglike;
    Vec Nu(1, nu());
    while(dloglike > eps){
      Estep();
      mvn->mle();
      MvtNuTF f(this);
      loglike = max_nd1(Nu, Target(f), dTarget(f));
      set_nu(Nu[0]);
      dloglike = loglike-old;
      old = loglike;
    }
  }

  double MVT::complete_data_loglike()const{
    double ans = mvn->loglike();
    ans+= wgt->loglike();
    return ans;
  }

  Vec MVT::sim()const{
    Vec ans = rmvn(mu().zero(), Sigma());
    double nu = this->nu();
    double w = rgamma(nu/2.0, nu/2.0);
    return mu() + ans/sqrt(w);
  }

}
