/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <Models/Glm/BinomialLogitModel.hpp>
#include <distributions.hpp>
#include <stats/logit.hpp>

namespace BOOM{

  typedef BinomialLogitModel BLM;
  typedef BinomialRegressionData BRD;
  typedef BinomialLogitLogLikelihood BLLL;


  BLLL::BinomialLogitLogLikelihood(const BLM *m)
      : m_(m)
  {}

  double BLLL::operator()(const Vector & beta)const{
    return m_->log_likelihood(beta, 0, 0);
  }
  double BLLL::operator()(const Vector & beta, Vector &g)const{
    return m_->log_likelihood(beta, &g, 0);
  }
  double BLLL::operator()(const Vector & beta, Vector &g, Matrix &H)const{
    return m_->log_likelihood(beta, &g, &H);
  }

  BLM::BinomialLogitModel(uint beta_dim, bool all)
      : ParamPolicy(new GlmCoefs(beta_dim, all)),
      log_alpha_(0)
      {}

  BLM::BinomialLogitModel(const Vector &beta)
      : ParamPolicy(new GlmCoefs(beta)),
      log_alpha_(0)
      {}

  BLM::BinomialLogitModel(const Matrix &X, const Vector &y, const Vector &n)
      : ParamPolicy(new GlmCoefs(X.ncol())),
      log_alpha_(0)
      {
        int nr = nrow(X);
        for(int i = 0; i < nr; ++i){
          uint yi = lround(y[i]);
          uint ni = lround(n[i]);
          NEW(BinomialRegressionData, dp)(yi, ni, X.row(i));
          add_data(dp);
        }
      }

  BLM::BinomialLogitModel(const BLM &rhs)
      : Model(rhs),
        MixtureComponent(rhs),
        GlmModel(rhs),
        NumOptModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        log_alpha_(rhs.log_alpha_) {}

  BLM* BinomialLogitModel::clone()const{
    return new BinomialLogitModel(*this);}

  double BLM::pdf(dPtr dp, bool logscale) const{
    return pdf(DAT(dp), logscale);}

  double BLM::pdf(const Data * dp, bool logscale) const{
    const BinomialRegressionData *rd =
        dynamic_cast<const BinomialRegressionData *>(dp);
    return logp(rd->y(), rd->n(), rd->x(), logscale);
  }

  double BLM::pdf(Ptr<BRD> dp, bool logscale) const{
    return logp(dp->y(), dp->n(), dp->x(), logscale);
  }

  double BLM::logp_1(bool y, const Vector &x, bool logscale)const{
    double btx = predict(x);
    double ans = -lope(btx);
    if(y) ans += btx;
    return logscale ? ans : exp(ans);
  }

  double BLM::logp(uint y, uint n, const Vector &x, bool logscale)const{
    double eta = predict(x);
    double p = logit_inv(eta);
    return dbinom(y, n, p, logscale);
  }

  double BLM::Loglike(const Vector &beta, Vector &g, Matrix &h, uint nd)const{
    if(nd>=2) return log_likelihood(beta, &g, &h);
    if(nd==1) return log_likelihood(beta, &g, 0);
    return log_likelihood(beta, 0, 0);
  }

  double BLM::log_likelihood(const Vector & beta, Vector *g, Matrix *h,
                             bool initialize_derivs)const{
    const BLM::DatasetType &data(dat());
    if (initialize_derivs) {
      if (g){
        g->resize(beta.size());
        *g=0;
        if (h) {
          h->resize(beta.size(), beta.size());
          *h=0;
        }
      }
    }
    double ans = 0;
    bool all_coefficients_included = (xdim() == beta.size());
    const Selector &inc(coef().inc());
    for(int i = 0; i < data.size(); ++i){
      // y and n had been defined as uint's but y-n*p was computing
      // -n, which overflowed
      int y = data[i]->y();
      int n = data[i]->n();
      const Vector & x(data[i]->x());
      Vector reduced_x;
      if (!all_coefficients_included) {
        reduced_x = inc.select(x);
      }
      ConstVectorView X(all_coefficients_included ? x : reduced_x);

      double eta = beta.dot(X) - log_alpha_;
      double p = logit_inv(eta);
      double loglike = dbinom(y, n, p, true);
      ans += loglike;
      if (g) {
        g->axpy(X, y-n*p);  // g += (y-n*p) * x;
        if (h) {
          h->add_outer(X, X, -n*p*(1-p)); // h += -npq * x x^T
        }
      }
    }
    return ans;
  }

  BLLL BLM::log_likelihood_tf()const{ return BLLL(this); }

  Spd BLM::xtx()const{
    const std::vector<Ptr<BinomialRegressionData> > & d(dat());
    uint n = d.size();
    uint p = d[0]->xdim();
    Spd ans(p);
    for(uint i=0; i<n; ++i){
      double n = d[i]->n();
      ans.add_outer(d[i]->x(), n, false);
    }
    ans.reflect();
    return ans;
  }

  void BLM::set_nonevent_sampling_prob(double alpha){
    if(alpha <=0 || alpha > 1){
      ostringstream err;
      err << "alpha (proportion of non-events retained in the data) "
          << "must be in (0,1]" << endl
          << "you set alpha = " << alpha << endl;
      throw_exception<std::runtime_error>(err.str());
    }
    log_alpha_ = std::log(alpha);
  }

  double BLM::log_alpha()const{return log_alpha_;}

}  // namespace BOOM
