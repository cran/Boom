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
#ifndef BOOM_BINOMIAL_LOGIT_MODEL_HPP_
#define BOOM_BINOMIAL_LOGIT_MODEL_HPP_
#include <Models/Glm/BinomialRegressionData.hpp>
#include <BOOM.hpp>
#include <TargetFun/TargetFun.hpp>
#include <numopt.hpp>
#include <Models/Glm/Glm.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>
namespace BOOM{

  class BinomialLogitModel;
  class BinomialLogitLogLikelihood
      : public d2TargetFun
  {
   public:
    BinomialLogitLogLikelihood(const BinomialLogitModel *m);
    double operator()(const Vec &beta)const;
    double operator()(const Vec &beta, Vec &g)const;
    double operator()(const Vec &beta, Vec &g, Mat &H)const;
   private:
    const BinomialLogitModel *m_;
  };

// logistic regression model with binned training data
  class BinomialLogitModel
      : public GlmModel,
        public NumOptModel,
        public ParamPolicy_1<GlmCoefs>,
        public IID_DataPolicy<BinomialRegressionData>,
        public PriorPolicy,
        public MixtureComponent
  {
   public:
    BinomialLogitModel(uint beta_dim, bool include_all=true);
    BinomialLogitModel(const Vec &beta);
    BinomialLogitModel(const Mat &X, const Vec &y, const Vec &n);
    BinomialLogitModel(const BinomialLogitModel &);
    BinomialLogitModel *clone()const;

    virtual GlmCoefs & coef(){return ParamPolicy::prm_ref();}
    virtual const GlmCoefs & coef()const{return ParamPolicy::prm_ref();}
    virtual Ptr<GlmCoefs> coef_prm(){return ParamPolicy::prm();}
    virtual const Ptr<GlmCoefs> coef_prm()const{return ParamPolicy::prm();}

    virtual double pdf(const Data * dp, bool logscale)const;
    virtual double pdf(dPtr dp, bool logscale)const;
    virtual double pdf(Ptr<BinomialRegressionData>, bool)const;
    virtual double logp(uint y, uint n, const Vec &x, bool logscale)const;
    virtual double logp_1(bool y, const Vec &x, bool logscale)const;

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    virtual double log_likelihood(const Vec &beta, Vec *g, Mat *h,
                                  bool initialize_derivs = true)const;

    BinomialLogitLogLikelihood log_likelihood_tf()const;

    virtual Spd xtx()const;

    // see comments in LogisticRegressionModel
    void set_nonevent_sampling_prob(double alpha);
    double log_alpha()const;

   private:
    double log_alpha_;  // see comments in logistic_regression_model
  };


}
#endif// BOOM_BINOMIAL_LOGIT_MODEL_HPP_
