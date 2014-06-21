/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include <Models/Glm/ProbitRegression.hpp>
#include <Models/Glm/RegressionModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <distributions.hpp>
#include <boost/bind.hpp>

namespace BOOM{

  typedef ProbitRegressionModel PRM;
  typedef BinaryRegressionData BRD;

  PRM::ProbitRegressionModel(const Vec &beta)
      : ParamPolicy(new GlmCoefs(beta))
  {}


  PRM::ProbitRegressionModel(const Mat &X, const Vec &y)
    : ParamPolicy(new GlmCoefs(ncol(X)))
  {
    int n = nrow(X);
    for(int i = 0; i < n; ++i){
      NEW(BinaryRegressionData, dp)(y[i]>.5,X.row(i));
      add_data(dp);
    }
  }

  PRM::ProbitRegressionModel(const ProbitRegressionModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GlmModel(rhs),
      NumOptModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs)
  {
  }

  PRM * PRM::clone()const{return new PRM(*this);}

  GlmCoefs & PRM::coef(){return ParamPolicy::prm_ref();}
  const GlmCoefs & PRM::coef()const{return ParamPolicy::prm_ref();}
  Ptr<GlmCoefs> PRM::coef_prm(){return ParamPolicy::prm();}
  const Ptr<GlmCoefs> PRM::coef_prm()const{return ParamPolicy::prm();}

  double PRM::pdf(dPtr dp, bool logscale)const{ return pdf(DAT(dp), logscale); }

  double PRM::pdf(Ptr<BinaryRegressionData> dp, bool logscale)const{
    return pdf(dp->y(), dp->x(), logscale);}

  double PRM::pdf(bool y, const Vec &x, bool logscale)const{
    double eta = predict(x);
    if(y) return pnorm(eta, 0, 1, true, logscale);
    return pnorm(eta, 0, 1, false, logscale);}

  double PRM::Loglike(Vec &g, Mat &h, uint nd)const{
    const Vec & b(Beta());
    if(nd==0) return log_likelihood(b, 0, 0);
    else if(nd==1) return log_likelihood(b,&g,0);
    return log_likelihood(b,&g,&h);
  }

  // see probit_loglike.tex for the calculus
  double PRM::log_likelihood(const Vec & beta, Vec * g, Mat * h, bool initialize_derivs)const{
    const PRM::DatasetType & data(dat());
    int n = data.size();
    if(initialize_derivs){
      if(g){
        *g=0;
        if(h) *h=0;
      }
    }
    double ans = 0;
    for(int i = 0; i < n; ++i){
      bool y = data[i]->y();
      const Vec & x(data[i]->x());
      double eta = beta.dot(x);
      double increment = pnorm(eta, 0, 1, y, true);
      ans += increment;
      if(g){
        double logp = y ? increment : pnorm(eta, 0, 1, true, true);
        double p = exp(logp);
        double q = 1-p;
        double v = p * q;
        double resid = (static_cast<double>(y)-p)/v;
        double phi = dnorm(eta);
        g->axpy(x, phi * resid);
        if(h){
          double pe = phi * resid;
          h->add_outer(x,x,-pe * (pe + eta));
        }
      }
    }
    return ans;
  }

  ProbitRegressionTarget PRM::log_likelihood_tf()const{
    ProbitRegressionTarget ans(this);
    return ans;
  }

  bool PRM::sim(const Vec &x)const{
    return runif() < pnorm(predict(x));
  }

  Ptr<BinaryRegressionData> PRM::sim()const{
    Vec x(xdim());
    x.randomize(boost::bind(&rnorm, 0, 1));
    bool y = this->sim(x);
    NEW(BinaryRegressionData, ans)(y,x);
    return ans;
  }

  ProbitRegressionTarget::ProbitRegressionTarget(const PRM * m)
      : m_(m)
  {}

  double ProbitRegressionTarget::operator()(const Vec &beta)const{
    Vec *g = 0;
    Mat *h = 0;
    return m_->log_likelihood(beta, g, h);
  }

  double ProbitRegressionTarget::operator()(const Vec &beta, Vec &g)const{
    Mat *h = 0;
    return m_->log_likelihood(beta, &g, h);
  }

  double ProbitRegressionTarget::operator()(const Vec &beta, Vec &g, Mat & h)const{
    return m_->log_likelihood(beta, &g, &h);
  }


}
