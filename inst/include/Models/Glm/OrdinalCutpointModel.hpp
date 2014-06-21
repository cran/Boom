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

#ifndef ORDINAL_CUTPOINT_MODEL_HPP
#define ORDINAL_CUTPOINT_MODEL_HPP

#include <Models/Glm/Glm.hpp>
#include <Models/CategoricalData.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <TargetFun/TargetFun.hpp>

// Model:  Y can be 0... M-1
// Pr(Y-m) = F(d[m]-btx) - F(d[m-1] - btx)
// where F is the link function and btx is "beta transpose x"
// d[] is the set of cutpoints with identifiability constraints:
// d[-1] = -infinity, d[0] = 0, d[M-1]=infinity

namespace BOOM{

  class OrdinalCutpointModel;

  class OrdinalCutpointBetaLogLikelihood
      : public TargetFun
  {
   public:
    OrdinalCutpointBetaLogLikelihood(const OrdinalCutpointModel *m_);
    double operator()(const Vec & beta)const;
   private:
    const OrdinalCutpointModel *m_;
  };

  class OrdinalCutpointDeltaLogLikelihood
      : public TargetFun
  {
   public:
    OrdinalCutpointDeltaLogLikelihood(const OrdinalCutpointModel *m_);
    double operator()(const Vec & delta)const;
   private:
    const OrdinalCutpointModel *m_;
  };

  class OrdinalCutpointModel:
    public ParamPolicy_2<GlmCoefs, VectorParams>,
    public IID_DataPolicy<OrdinalRegressionData>,
    public PriorPolicy,
    public GlmModel,
    public NumOptModel
  {

  public:
    OrdinalCutpointModel(const Vec &beta, const Vec & delta);
    OrdinalCutpointModel(const Vec &beta,
                         const Selector &Inc,
                         const Vec & delta);
    OrdinalCutpointModel(const Selector &Inc, uint Maxscore);
    OrdinalCutpointModel(const Mat &X, const Vec &y);
    OrdinalCutpointModel(const OrdinalCutpointModel &rhs);

    virtual OrdinalCutpointModel *clone()const=0;

    // link_inv(eta) = probability
    // link(prob) = eta
    virtual double link_inv(double)const=0;  // logit or probit
    virtual double dlink_inv(double)const=0; // derivative of link_inv

    virtual GlmCoefs & coef();
    virtual const GlmCoefs & coef()const;
    virtual Ptr<GlmCoefs> coef_prm();
    virtual const Ptr<GlmCoefs> coef_prm()const;

    Ptr<VectorParams> Delta_prm();
    const Ptr<VectorParams> Delta_prm()const;

    // inherits [Bb]eta()/set_[Bb]eta() from GlmModel
    double delta(uint)const; // delta[0] = - infinity, delta[1] = 0
    const Vec & delta()const;
    void set_delta(const Vec &d);
    bool check_delta(const Vec & Delta)const;  // if Delta satisfies constraint

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    double log_likelihood(const Vec & beta, const Vec & delta)const;
    OrdinalCutpointBetaLogLikelihood beta_log_likelihood()const;
    OrdinalCutpointDeltaLogLikelihood delta_log_likelihood()const;

    void initialize_params();
    void initialize_params(const Vec &counts);

    Vec CDF(const Vec &x)const; // Pr(Y<y)

    virtual double pdf(dPtr, bool)const;
    double pdf(Ptr<OrdinalRegressionData>, bool)const;
    double pdf(const OrdinalData &y, const Vec &x, bool logscale)const;
    double pdf(uint y, const Vec &x, bool logscale)const;

    uint maxscore()const; // maximum possible score allowed

    Ptr<OrdinalRegressionData> sim();

  private:
    // interface is complicated
    double bd_loglike(Vec & gbeta, Vec &gdelta, Mat & Hbeta, Mat &Hdelta,
		      Mat & Hbd, uint nd, bool b_derivs, bool d_derivs) const;
    Ptr<CatKey> simulation_key_;
    virtual double simulate_latent_variable()const=0;  // from the link distribution
  };

  } // closes namespace BOOM

#endif// ORDINAL_CUTPOINT_MODEL_HPP
