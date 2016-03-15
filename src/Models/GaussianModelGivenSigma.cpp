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
#include <Models/GaussianModelGivenSigma.hpp>
#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <distributions.hpp>
#include <cmath>

namespace BOOM{

  typedef GaussianModelGivenSigma GMGS;

  GaussianModelGivenSigma::GaussianModelGivenSigma(Ptr<UnivParams> sigsq,
                                                   double mu0, double kappa)
    : Model(),
      ParamPolicy(new UnivParams(mu0), new UnivParams(kappa)),
      PriorPolicy(),
      DataPolicy(new GaussianSuf()),
      sigsq_(sigsq)
  { }

  GMGS * GMGS::clone()const{return new GMGS(*this);}

  Ptr<UnivParams> GMGS::Mu_prm(){ return prm1(); }
  Ptr<UnivParams> GMGS::Kappa_prm(){ return prm2(); }
  const Ptr<UnivParams> GMGS::Mu_prm()const{ return prm1(); }
  const Ptr<UnivParams> GMGS::Kappa_prm()const{ return prm2(); }

  void GMGS::set_params(double mu, double kappa){
    set_mu(mu); set_kappa(kappa); }

  void GMGS::set_sigsq(Ptr<UnivParams> s){
    assert(s->value()>0);
    sigsq_ = s; }

  void GMGS::set_mu(double m){ Mu_prm()->set(m); }
  void GMGS::set_kappa(double s){ Kappa_prm()->set(s); }

  double GMGS::ybar()const{return suf()->ybar();}
  double GMGS::sample_var()const{return suf()->sample_var();}

  double GMGS::mu()const{return Mu_prm()->value();}
  double GMGS::kappa()const{return Kappa_prm()->value();}
  double GMGS::sigsq()const{return sigsq_->value();}

  void GMGS::mle(){
    double n = suf()->n();
    double m = n < 1 ? 0 : ybar();
    double sigma_hat_squared = sample_var()*(n-1)/n;
    double kappa = n<=1 ? 1.0 : sigsq() / sigma_hat_squared;
    set_params(m,kappa);
  }

  double GMGS::Logp(double x, double &g, double &h, uint nd)const{
    double m = mu();
    double v = var();
    double ans = dnorm(x, m, sqrt(v), 1);
    if(nd>0) g = -(x-m)/v;
    if(nd>1) h = -1.0/v;
    return ans;
  }

  double GMGS::Logp(const Vector &x, Vector &g, Matrix &h, uint nd)const{
    double X=x[0];
    double G(0),H(0);
    double ans = Logp(X,G,H,nd);
    if(nd>0) g[0]=G;
    if(nd>1) h(0,0)=H;
    return ans;
  }

  double GMGS::Loglike(const Vector &mu_kappa, Vector &g, Matrix &h, uint nd) const {
    if (mu_kappa.size() != 2 || mu_kappa[1] <= 0) {
      report_error("Illegal argument passed to GaussianModelGivenSigma::Loglike.");
    }
    double sigsq = this->sigsq();
    if(sigsq<0) return BOOM::negative_infinity();

    double mu = mu_kappa[0];
    const double log2pi = 1.8378770664093453;
    double n = suf()->n();
    double sumsq = suf()->sumsq();
    double sum = suf()->sum();
    double SS = (sumsq + ( -2*sum + n*mu)*mu);
    double k = mu_kappa[1];
    double ans = -0.5*(n*(log2pi + log(sigsq) - log(k) + k*SS/sigsq));

    if(nd>0){
      g[0] = k*(sum-n*mu)/sigsq;
      g[1] = 0.5*(n/k - SS/sigsq);
      if(nd>1){
	h(0,0) = -n*k/sigsq;
	h(1,0) = h(0,1) = (sum-n*mu)/sigsq;
	h(1,1) = -0.5/(k*k);}}

    return ans;
  }

  double GMGS::var()const{return sigsq()/kappa();}
  double GMGS::sim()const{ return rnorm(mu(), sqrt(var()));}

  void GMGS::add_data_raw(double x){
    NEW(DoubleData, dp)(x);
    this->add_data(dp);
  }

}
