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

#ifndef PROBIT_REGRESSION_HPP
#define PROBIT_REGRESSION_HPP

#include <BOOM.hpp>
#include <Models/Glm/Glm.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <numopt.hpp>
#include <TargetFun/TargetFun.hpp>

namespace BOOM{

  class ProbitRegressionModel;

  class ProbitRegressionTarget : public d2TargetFun{
   public:
    ProbitRegressionTarget(const ProbitRegressionModel *m);
    double operator()(const Vec &beta)const;
    double operator()(const Vec &beta, Vec &g)const;
    double operator()(const Vec &beta, Vec &g, Mat &h)const;
   private:
    const ProbitRegressionModel *m_;
  };

  class ProbitRegressionModel
      : public GlmModel,
        public NumOptModel,
        public ParamPolicy_1<GlmCoefs>,
        public IID_DataPolicy<BinaryRegressionData>,
        public PriorPolicy
  {
  public:
    ProbitRegressionModel(const Vec &beta);
    ProbitRegressionModel(const Mat &X, const Vec &y);
    ProbitRegressionModel(const ProbitRegressionModel &);
    ProbitRegressionModel *clone()const;

    virtual GlmCoefs & coef();
    virtual const GlmCoefs & coef()const;
    virtual Ptr<GlmCoefs> coef_prm();
    virtual const Ptr<GlmCoefs> coef_prm()const;

    virtual double pdf(dPtr, bool)const;
    virtual double pdf(Ptr<BinaryRegressionData>, bool)const;
    virtual double pdf(bool y, const Vec &x, bool logscale)const;

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;

    // call with *g=0 if you don't want any derivatives.  Call with
    // *g!=0 and *h=0 if you only want first derivatives.
    // if(initialize_derivs) then *g and *h will be set to zero.
    // Otherwise they will be incremented
    double log_likelihood(const Vec & beta, Vec *g, Mat *h,
                          bool initialize_derivs = true)const;
    ProbitRegressionTarget log_likelihood_tf()const;

    bool sim(const Vec &x)const;
    Ptr<BinaryRegressionData> sim()const;

  };

}// ends namespace BOOM

#endif //PROBIT_REGRESSION_HPP
