/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MVN_GIVEN_X_AND_SIGMA_HPP
#define BOOM_MVN_GIVEN_X_AND_SIGMA_HPP

#include <Models/ParamTypes.hpp>
#include <Models/MvnBase.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

#include <Models/Glm/Glm.hpp>
#include <Models/Glm/RegressionModel.hpp>

namespace BOOM{

  class MvnGivenXandSigma
    : public MvnBase,
      public ParamPolicy_2<VectorParams, UnivParams>,
      public IID_DataPolicy<GlmCoefs>,
      public PriorPolicy
  {
    // model is that beta | X,sigsq ~ N(b, kappa/n * V)
    // where V^{-1}  = Lambda + (1-w) XTX + w Diag(XTX)
    // with Lambda a diagonal matrix, XTX the cross product matrix
    // from the regression model, and Diag(XTX) the diagonal matrix
    // with elements from the diagonal of XTX.

    // in this model kappa is interpreted as the prior sample size

  public:
    MvnGivenXandSigma(RegressionModel *, Ptr<VectorParams> Mu,
		      Ptr<UnivParams> prior_ss, double diag_wgt=0);
    MvnGivenXandSigma(RegressionModel *, Ptr<VectorParams> Mu,
		      Ptr<UnivParams> prior_ss,
		      const Vector & Lambda, double diag_wgt=0);
    MvnGivenXandSigma(const MvnGivenXandSigma &rhs);
    MvnGivenXandSigma * clone() const override;

    const Vector & mu()const override;
    const SpdMatrix & Sigma()const override;
    const SpdMatrix & siginv()const override;
    double ldsi()const override;

    double prior_sample_size()const;
    virtual double pdf(Ptr<Data>, bool logscale)const;

    const Ptr<VectorParams> Mu_prm()const;
    const Ptr<UnivParams> Kappa_prm()const;
    Ptr<VectorParams> Mu_prm();
    Ptr<UnivParams> Kappa_prm();
    double diagonal_weight()const;

    Vector sim()const override;
  private:
    RegressionModel *  mod_;
    mutable Ptr<SpdParams> ivar_;
    Vector Lambda_;
    double diagonal_weight_;

    mutable bool current_;
    void set_ivar()const; // logical constness
  };
}
#endif// BOOM_MVN_GIVEN_X_AND_SIGMA_HPP
