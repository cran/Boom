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

#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <BOOM.hpp>
#include <TargetFun/TargetFun.hpp>
#include <numopt.hpp>
#include <Models/Glm/Glm.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>

namespace BOOM{

  class LogisticRegressionModel;

  class LogitLogLikelihood : public d2TargetFun{
   public:
    LogitLogLikelihood(const LogisticRegressionModel *m);
    double operator()(const Vec & beta)const;
    double operator()(const Vec & beta, Vec &g)const;
    double operator()(const Vec & beta, Vec &g, Mat &h)const;
   private:
    const LogisticRegressionModel *m_;
  };

  class LogisticRegressionModel
      : public GlmModel,
        public NumOptModel,
        public MixtureComponent,
        public ParamPolicy_1<GlmCoefs>,
        public IID_DataPolicy<BinaryRegressionData>,
        public PriorPolicy
  {
  public:
    LogisticRegressionModel(uint beta_dim, bool include_all=true);
    LogisticRegressionModel(const Vec &beta);
    LogisticRegressionModel(const Mat &X, const Vec &y, bool add_int);
    LogisticRegressionModel(const LogisticRegressionModel &);
    LogisticRegressionModel *clone()const;

    virtual GlmCoefs &coef(){return ParamPolicy::prm_ref();}
    virtual const GlmCoefs &coef()const{return ParamPolicy::prm_ref();}
    virtual Ptr<GlmCoefs> coef_prm(){return ParamPolicy::prm();}
    virtual const Ptr<GlmCoefs> coef_prm()const{return ParamPolicy::prm();}

    virtual double pdf(dPtr dp, bool logscale)const;
    virtual double pdf(const Data * dp, bool logscale)const;
    double logp(bool y, const Vec &x)const;

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    virtual double log_likelihood(const Vec &beta, Vec *g, Mat *h,
                                  bool initialize_derivs = true)const;

    LogitLogLikelihood log_likelihood_tf()const;

    virtual Spd xtx()const;

    // when modeling rare events it can be convenient to retain all
    // the events and 100 * alpha% of the non-events.
    void set_nonevent_sampling_prob(double alpha);
    double log_alpha()const;
   private:
    double log_alpha_;  // alpha is the probability that a 'zero'
                        // (non-event) is retained in the data.  It is
                        // assumed that the data retains all the
                        // events and 100 alpha% of the non-events
  };

  class MvnBase;
  // Logistic regression specialized to be an EM mixture component
  class LogitEMC
    : public LogisticRegressionModel,
      public EmMixtureComponent
  {
  public:
    LogitEMC(uint beta_dim, bool all=true);
    LogitEMC(const Vec &beta);
    LogitEMC * clone()const;
    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    virtual void add_mixture_data(Ptr<Data>, double prob);
    virtual void clear_data();
    virtual double pdf(Ptr<Data> dp, bool logscale)const{
      return LogisticRegressionModel::pdf(dp, logscale);}
    virtual double pdf(const Data *dp, bool logscale)const{
      return LogisticRegressionModel::pdf(dp, logscale);}
    void set_prior(Ptr<MvnBase>);
    void find_posterior_mode();
    virtual Spd xtx()const;    // incorporates probs
  private:
    Vec probs_;
    Ptr<MvnBase> pri_;
  };


}// ends namespace BOOM

#endif //LOGISTIC_REGRESSION_HPP
