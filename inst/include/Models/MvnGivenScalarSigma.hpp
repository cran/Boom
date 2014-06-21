/*
  Copyright (C) 2010 Steven L. Scott

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

#ifndef BOOM_MVN_GIVEN_SCALAR_SIGMA_HPP_
#define BOOM_MVN_GIVEN_SCALAR_SIGMA_HPP_

#include <Models/MvnBase.hpp>
#include <Models/SpdData.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  // This model is intended for use as a conditional prior
  // distribution for regression coefficients in "least squares"
  // regression problems.  The model is
  //
  // beta | sigsq ~ N(b, sigsq * Omega)
  //
  // Omega inverse will typically be some multiple of the "XTX" cross
  // product matrix in the regression model, so the constructors take
  // the inverse of Omega as an argument.  Omega is a fixed constant in
  // this model, which might make it a poor fit for hierarchical models
  // where the degree of shrinkage is to be learned across groups.
  class MvnGivenScalarSigma
      : public MvnBase,
        public LoglikeModel,
        public ParamPolicy_1<VectorParams>,
        public SufstatDataPolicy<VectorData, MvnSuf>,
        public PriorPolicy
  {
   public:
    MvnGivenScalarSigma(const Spd &ominv, Ptr<UnivParams> sigsq);
    MvnGivenScalarSigma(const Vec &mean,
                        const Spd &ominv,
                        Ptr<UnivParams> sigsq);

    MvnGivenScalarSigma(const MvnGivenScalarSigma & rhs);
    virtual MvnGivenScalarSigma * clone()const;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm()const;

    virtual uint dim()const;
    virtual const Vec & mu() const;

    // Sigma refers to the actual variance matrix of beta given sigma
    // and Omega, i.e. Omega * sigsq.  siginv and ldsi refer to its
    // inverse and the log of the determinant of its inverse.
    virtual const Spd & Sigma()const;
    virtual const Spd & siginv()const;
    virtual double ldsi()const;

    // Omega refers to the proportional variance matrix of beta
    // (i.e. not multiplied by sigsq).  ominv and ldoi refer to the
    // inverse of this matrix and the log of the determinant of the
    // inverse.
    const Spd & Omega()const;
    const Spd & ominv()const;
    double ldoi()const;

    void set_mu(const Vec &);
    void mle();
    double loglike()const;
    double pdf(Ptr<Data>, bool)const;
   private:
    // sigsq_ is a pointer to the residual variance parameter, e.g. in
    // a regression model.
    Ptr<UnivParams> sigsq_;

    // ominv_ is stored as SpdParams instead of as a raw Spd because
    // SpdParams keeps track of the matrix, its inverse, and its log
    // determinant.
    SpdData omega_;

    // The following is workspace used to comply with the
    // return-by-reference interface promised by MvnBase for Sigma(),
    // siginv(), and ldsi().
    mutable Spd wsp_;
  };

}
#endif  // BOOM_MVN_GIVEN_SCALAR_SIGMA_HPP_
