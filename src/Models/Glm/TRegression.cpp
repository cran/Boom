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

#include <Models/Glm/TRegression.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/ScaledChisqModel.hpp>
#include <Models/Glm/WeightedRegressionModel.hpp>

#include <LinAlg/Types.hpp>
#include <distributions.hpp>
#include <numopt.hpp>

#include <iomanip>
#include <cmath>


using std::setw;
using std::setprecision;

namespace BOOM{

  typedef WeightedRegressionData WRD;
  typedef WeightedRegressionModel WRM;
  typedef ScaledChisqModel SCM;
  typedef DoubleData DD;
  typedef TRegressionModel TRM;

  void TRM::setup_params(){
    ParamPolicy::add_model(wreg_);
    ParamPolicy::add_model(wgt_);
  }

  TRM::TRegressionModel(uint p)
    : wreg_(new WRM(p)),
      wgt_(new SCM)
  {
    setup_params();
  }

  TRM::TRegressionModel(const Vector &b, double Sigma, double nu)
    : wreg_(new WRM(b,Sigma)),
      wgt_(new SCM(nu))
  {
    setup_params();
  }


  TRM::TRegressionModel(const TRegressionModel &rhs)
    : Model(rhs),
      GlmModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      NumOptModel(rhs),
      LatentVariableModel(rhs),
      wreg_(rhs.wreg_->clone()),
      wgt_(rhs.wgt_->clone())
  {
    setup_params();
  }


  TRM::TRegressionModel(const DatasetType &d, bool all)
    : GlmModel(),
      ParamPolicy(),
      DataPolicy(),
      PriorPolicy(),
      LatentVariableModel(),
      wreg_(new WRM(all ? d[0]->xdim() : 1)),
      wgt_(new SCM)
  {
    setup_params();
    uint n = d.size();
    for(uint i=0; i<n; ++i) add_data(d[i]);
    mle();
  }


  TRM * TRM::clone()const{ return new TRM(*this);}

  GlmCoefs & TRM::coef(){return wreg_->coef();}
  const GlmCoefs & TRM::coef()const{return wreg_->coef();}
  Ptr<GlmCoefs> TRM::coef_prm(){return wreg_->coef_prm();}
  const Ptr<GlmCoefs> TRM::coef_prm()const{return wreg_->coef_prm();}
  Ptr<UnivParams> TRM::Sigsq_prm(){return wreg_->Sigsq_prm();}
  const Ptr<UnivParams> TRM::Sigsq_prm()const{return wreg_->Sigsq_prm();}
  Ptr<UnivParams> TRM::Nu_prm(){return wgt_->Nu_prm();}
  const Ptr<UnivParams> TRM::Nu_prm()const{return wgt_->Nu_prm();}
  void TRM::set_sigsq(double s2){wreg_->set_sigsq(s2);}
  void TRM::set_nu(double Nu){wgt_->set_nu(Nu);}

  double TRM::Loglike(const Vector &beta_sigsq_nu,
                      Vector &g, Matrix &h, uint nd)const{
    double nu = beta_sigsq_nu.back();
    double sigsq = beta_sigsq_nu[beta_sigsq_nu.size() - 2];
    const Selector &inclusion_indicators(coef().inc());
    int beta_dim = inclusion_indicators.nvars();
    const Vector beta(ConstVectorView(beta_sigsq_nu, 0, beta_dim));
    double ans=0;
    if(nd>0){
      g=0;
      h=0;
    }

    for(uint i=0; i < dat().size(); ++i){
      const Vector X = coef().inc().select((dat())[i]->x());
      const double yhat = beta.dot(X);
      const double y = (dat())[i]->y();
      ans+= dstudent(y, yhat, sigsq, nu, true);
      if(nd>0){
 	double e = y-yhat;
 	double esq_ns =e*e/(nu*sigsq);
 	double frac = esq_ns/(1+esq_ns);

 	Vector gbeta = ((nu+1)*frac/e) *X;

 	Vector gsignu(2);
 	gsignu[0] = -1/(2*sigsq);
 	gsignu[0]*= (1-(nu+1)*frac);

 	gsignu[1] = .5*(digamma((nu+1)/2)- digamma(nu/2) - 1.0/nu
			-log(1+esq_ns) + frac*(nu+1)/nu);
 	g += concat(gbeta, gsignu);
 	if(nd>1){
          report_error(
              "second derivatives of TRegression are not yet implemented.");
 	  double esq = e*e;
 	  double sn = sigsq*nu;
 	  double esp = esq + sn;

 	  Matrix hbb = X.outer()* ((nu+1)*( (esq -sn)/esp));
 	  Vector hbs = (-e*(nu+1)*nu/pow(esp, 2)) * X;
 	  Vector hbn = ((e/esp)*(1-(nu+1)*sigsq/esp)) * X;}}}
    return ans;
  }


  class TrmNuTF{
  public:
    TrmNuTF(TRegressionModel *Mod) : mod(Mod){}
    TrmNuTF * clone()const{return new TrmNuTF(*this);}
    double operator()(const Vector &Nu)const;
    double operator()(const Vector &Nu, Vector &g)const;
  private:
    double Loglike(const Vector &Nu, Vector &g, uint nd)const;
    TRegressionModel *mod;
  };

  double TrmNuTF::operator()(const Vector &Nu)const{
    Vector g;
    return Loglike(Nu, g, 0);}
  double TrmNuTF::operator()(const Vector &Nu, Vector &g)const{
    return Loglike(Nu,g,1);}

  double TrmNuTF::Loglike(const Vector &Nu, Vector &g, uint nd)const{
    const std::vector<Ptr<RegressionData> > & dat(mod->dat());
    uint n = dat.size();
    double nu = Nu[0];

    double nh = .5*(nu+1);
    double logsig = log(mod->sigma());
    double lognu = log(nu);
    const double logpi= 1.1447298858494;
    double ans = lgamma(nh)-lgamma(nu/2)+ (nh-.5)*lognu - logsig - .5*logpi;
    ans *=n;

    if(nd>0){
      g[0] = .5*digamma(nh) - .5*digamma(nu/2) + (nh-.5)/nu + .5*lognu;
      g[0]*=n;
    }

    for(uint i=0; i<n; ++i){
      Ptr<RegressionData> dp = dat[i];
      double err = dp->y() - mod->predict(dp->x());
      double dsq = err*err/mod->sigsq();
      double lnpd = log(nu+dsq);
      ans -= nh*lnpd;
      if(nd>0) g[0] -=  nh/(nu+dsq) + .5*lnpd;
    }

    return ans;
  }


  void TRM::mle(){
    const double eps = 1e-5;
    double dloglike= eps+1;
    double loglike = this->loglike(vectorize_params());
    double old = loglike;
    Vector Nu(1, nu());
    while(dloglike > eps){
      EStep();
      wreg_->mle();
      TrmNuTF f(this);
      loglike = max_nd1(Nu, Target(f), dTarget(f));
      set_nu(Nu[0]);
      dloglike = loglike-old;
      old = loglike;
    }
  }

  double TRM::complete_data_loglike()const{
    Vector g;
    Matrix h;
    return complete_data_Loglike(g,h,0);
  }
  double TRM::complete_data_Loglike(Vector &g, Matrix &h, uint nd)const{

    uint p = Beta().size();

    Vector g_reg, g_wgt;
    Matrix h_reg, h_wgt;
    if(nd>0){
      g_reg.resize(p+1);
      g_wgt.resize(1);

      if(nd>1){
	h_reg.resize(p+1, p+1);
	h_wgt.resize(1,1);}}

    Vector beta_sigsq = Beta();
    beta_sigsq.push_back(sigsq());
    double ans = wreg_->Loglike(
        beta_sigsq,
        g_reg,
        h_reg,nd);

    Vector nu_vector(1, nu());
    ans += wgt_->Loglike(nu_vector, g_wgt, h_wgt, nd);

    if(nd>0){
      g = concat(g_reg, g_wgt);
      if(nd>1){
	h = block_diagonal(h_reg, h_wgt);}}
    return ans;}


  void TRM::impute_latent_data(RNG &rng){
    Impute(true, rng);
  }

  void TRM::EStep(){Impute(false, GlobalRng::rng);}

  void TRM::Impute(bool draw, RNG &rng){
    wreg_->suf()->clear();
    wgt_->suf()->clear();

    double nu2 = nu()/2.0;
    double df2 = nu2 + .5;
    double sigsq2 = sigsq()*2.0;

    std::vector<Ptr<WeightedRegressionData> > &
      regdat(wreg_->dat());

    for(uint i=0; i<regdat.size(); ++i){
      Ptr<WeightedRegressionData> dp = regdat[i];
      Ptr<DoubleData> wp = dp->WeightPtr();
      double err = dp->y() - predict(dp->x());
      double ss2 = err*err/sigsq2 + nu2;
      double w = draw ? df2/ss2 : rgamma_mt(rng, df2, ss2);
      dp->set_weight(w);
      wreg_->suf()->update(dp);
      wgt_->suf()->update(wp);
    }
  }

  void TRM::initialize_params(){

  }

  double TRM::pdf(Ptr<TRM::DataType> dp, bool logscale)const{
    double yhat = predict(dp->x());
    return dstudent(dp->y(), yhat, sigma(), nu(), logscale); }
  double TRM::pdf(Ptr<Data> dp, bool logscale)const{
    return pdf(dp.dcast<DataType>(), logscale);}


  const double & TRM::sigsq()const{return Sigsq_prm()->value();}
  double TRM::sigma()const{return sqrt(sigsq());}
  const double & TRM::nu()const{ return wgt_->nu();}


  Ptr<RegressionData>  TRM::simdat()const{
    uint p = Beta().size();
    Vector x(p);
    for(uint i=0; i<p; ++i) x[i] = rnorm();
    return simdat(x);
  }

  Ptr<RegressionData> TRM::simdat(const Vector &x)const{
    double nu = this->nu();
    double w = rgamma(nu/2, nu/2);
    double yhat = predict(x);
    double z = rnorm(0, sigma()/sqrt(w));
    double y = yhat +z;
    NEW(RegressionData, d)(y,x);
    return d;
  }


  void TRM::add_data(Ptr<RegressionData> dp){
    NEW(DoubleData, w)(1.0);
    NEW(WeightedRegressionData, wrd)(dp, w);
    wreg_->add_data(wrd);
    wgt_->add_data(w);
  }

  void TRM::add_data(Ptr<Data> dp){add_data(DAT(dp));}
}
