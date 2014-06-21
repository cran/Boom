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

#include <Models/Glm/LogisticRegressionModel.hpp>
#include <Models/Glm/PosteriorSamplers/LogitSampler.hpp>
#include <stats/logit.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cpputil/lse.hpp>
#include <Models/MvnBase.hpp>
#include <numopt.hpp>
#include <TargetFun/LogPost.hpp>

namespace BOOM{

  LogisticRegressionModel::LogisticRegressionModel(uint beta_dim, bool all)
      : ParamPolicy(new GlmCoefs(beta_dim, all)),
        log_alpha_(0)
  {}

  LogisticRegressionModel::LogisticRegressionModel(const Vec &beta)
      : ParamPolicy(new GlmCoefs(beta)),
        log_alpha_(0)
  {}


  LogisticRegressionModel::LogisticRegressionModel
  (const Mat &X, const Vec &y, bool add_int)
      : ParamPolicy(new GlmCoefs(X.ncol())),
        log_alpha_(0)
  {
    int n = nrow(X);
    for(int i = 0; i < n; ++i){
      NEW(BinaryRegressionData, dp)(y[i]>.5,X.row(i));
      add_data(dp);
    }
  }


  LogisticRegressionModel::LogisticRegressionModel
  (const LogisticRegressionModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GlmModel(rhs),
      NumOptModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      log_alpha_(rhs.log_alpha_)
  {}

  LogisticRegressionModel* LogisticRegressionModel::clone()const{
    return new LogisticRegressionModel(*this);}

  typedef LogisticRegressionModel LRM;
  typedef BinaryRegressionData BRD;

  double LRM::pdf(dPtr dp, bool logscale) const{
    Ptr<BRD> d = DAT(dp);
    double ans= logp(d->y(), d->x());
    return logscale ? ans : exp(ans);
  }

  double LRM::pdf(const Data * dp, bool logscale) const{
    const BRD * d = DAT(dp);
    double ans= logp(d->y(), d->x());
    return logscale ? ans : exp(ans);
  }

  double LRM::logp(bool y, const Vec &x)const{
    double btx = predict(x);
    double ans = -lope(btx);
    if(y) ans += btx;
    return ans;
  }

  double LRM::Loglike(Vec &g, Mat &h, uint nd)const{
    if(nd>=2) return log_likelihood(included_coefficients(), &g, &h);
    if(nd==1) return log_likelihood(included_coefficients(), &g, 0);
    return log_likelihood(included_coefficients(), 0, 0);
  }

  double LRM::log_likelihood(const Vec & beta, Vec *g, Mat *h,
                             bool initialize_derivs)const{
    const LRM::DatasetType &data(dat());
    if(initialize_derivs){
      if(g){
        g->resize(beta.size());
        *g=0;
        if(h){
          h->resize(beta.size(), beta.size());
          *h=0;}}}

    double ans = 0;
    int n = data.size();
    bool all_coefficients_included = coef().nvars() == xdim();
    const Selector &inc(coef().inc());
    for(int i = 0; i < n; ++i){
      bool y = data[i]->y();
      const Vec & x(data[i]->x());
      double eta = predict(x) + log_alpha_;
      double loglike = plogis(eta, 0, 1, y, true);
      ans += loglike;
      if(g){
        double logp = y ? loglike : plogis(eta, 0, 1, true, true);
        double p = exp(logp);
        if (all_coefficients_included) {
          *g += (y-p) * x;
          if(h){
            h->add_outer(x,x, -p*(1-p));
          }
        } else {
          Vector reduced_x = inc.select(x);
          *g += (y - p) * reduced_x;
          if (h) {
            h->add_outer(reduced_x, reduced_x, -p * (1 - p));
          }
        }
      }
    }
    return ans;
  }

  LogitLogLikelihood LRM::log_likelihood_tf()const{
    return LogitLogLikelihood(this);
  }

  Spd LRM::xtx()const{
    const std::vector<Ptr<BinaryRegressionData> > & d(dat());
    uint n = d.size();
    uint p = d[0]->xdim();
    Spd ans(p);
    for(uint i=0; i<n; ++i) ans.add_outer(d[i]->x(), 1.0, false);
    ans.reflect();
    return ans;
  }

  void LRM::set_nonevent_sampling_prob(double alpha){
    if(alpha <=0 || alpha > 1){
      ostringstream err;
      err << "alpha (proportion of non-events retained in the data) "
          << "must be in (0,1]" << endl
          << "you set alpha = " << alpha << endl;
      report_error(err.str());
    }
    log_alpha_ = std::log(alpha);
  }

  double LRM::log_alpha()const{return log_alpha_;}

  //______________________________________________________________________

  typedef LogitLogLikelihood LLL;
  LLL::LogitLogLikelihood(const LRM *m)
      : m_(m)
  {}

  double LLL::operator()(const Vec &b)const{
    return m_->log_likelihood(b, 0, 0);
  }
  double LLL::operator()(const Vec &b, Vec &g)const{
    return m_->log_likelihood(b, &g, 0);
  }
  double LLL::operator()(const Vec &b, Vec &g, Mat &h)const{
    return m_->log_likelihood(b, &g, &h);
  }

  //______________________________________________________________________

  LogitEMC::LogitEMC(uint beta_dim, bool all)
    : LRM(beta_dim, all)
  {}

  LogitEMC::LogitEMC(const Vec &beta)
    : LRM(beta)
  {}

  LogitEMC * LogitEMC::clone()const{return new LogitEMC(*this);}

  //----------------------------------------------------------------------

  double logit_loglike_1(const Vec & beta, bool y, const Vec &x,
			 Vec *g, Mat *h, double mix_wgt){
    double eta = x.dot(beta);
    double lognc = lse2(0, eta);
    double ans = y?  eta : 0;
    ans -= lognc;
    if(g){
      double p = exp(eta-lognc);
      g->axpy(x, mix_wgt* (y-p));
      if(h){
	double q = 1-p;
	h->add_outer( x,x, -mix_wgt * p*q);}}
    return mix_wgt * ans;
  }

  //----------------------------------------------------------------------
  double LogitEMC::Loglike(Vec &g, Mat &h, uint nd)const{
    uint n = probs_.size();
    if(n==0) return LRM::Loglike(g,h,nd);

    const DatasetType &d(dat());
    if(d.size()!=n){
      ostringstream err;
      err << "There is a mismatch between the data vector and the vector "
	  << "of mixing weights in LogitEMC::Loglike." << endl;
      report_error(err.str());
    }

    Vec * gp=0;
    Mat * hp=0;
    if(nd>0){
      g=0;
      gp = &g;
      if(nd>1){
	h=0;
	hp = &h;}}

    const Selector & inc(coef().inc());
    Vec b(this->included_coefficients());

    double ans=0;
    for(uint i=0; i<n; ++i){
      double w = probs_[i];
      Vec x = inc.select(d[i]->x());
      bool y = d[i]->y();
      ans += logit_loglike_1(b, y, x, gp, hp, w);
    }
    return ans;
  }

  void LogitEMC::add_mixture_data(Ptr<Data> dp, double prob){
    LRM::add_data(dp);
    probs_.push_back(prob);
  }

  void LogitEMC::clear_data(){
    LRM::clear_data();
    probs_.clear();
  }

  void LogitEMC::set_prior(Ptr<MvnBase> pri){
    pri_ = pri;
    NEW(LogitSampler, sam)(this, pri);
    set_method(sam);
  }

  Spd LogitEMC::xtx()const{
    uint n = probs_.size();
    if(n==0) return LRM::xtx();
    const std::vector<Ptr<BinaryRegressionData> > & d(dat());
    assert(d.size()==n);
    uint p = d[0]->xdim();
    Spd ans(p);
    for(uint i=0; i<n; ++i){
      ans.add_outer(d[i]->x(), probs_[i], false);
    }
    ans.reflect();
    return ans;
  }

  void LogitEMC::find_posterior_mode(){
    if(!pri_){
      ostringstream err;
      err << "Logit_EMC cannot find posterior mode.  "
	  << "No prior is set." << endl;
      report_error(err.str());
    }

    d2LoglikeTF loglike(this);
    d2LogPostTF logpost(loglike, pri_);
    Vec b = this->Beta();
    uint dim = b.size();
    Vec g(dim);
    Mat h(dim,dim);
    b = max_nd2(b, g, h, Target(logpost), dTarget(logpost),
                d2Target(logpost), 1e-5);
    this->set_Beta(b);
  }

}
