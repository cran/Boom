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

#ifndef POISSON_REGRESSION_MODEL_HPP
#define POISSON_REGRESSION_MODEL_HPP

#include <Models/Glm/Glm.hpp>
#include <Models/Glm/PoissonRegressionData.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/ModelTypes.hpp>

namespace BOOM {

  // A PoissonRegressionModel describes a non-negative integer
  // response y ~ Poisson(E exp(beta*x)), where E is an exposure.
  class PoissonRegressionModel
      : public GlmModel,
        public NumOptModel,
        public MixtureComponent,
        public ParamPolicy_1<GlmCoefs>,
        public IID_DataPolicy<PoissonRegressionData>,
        public PriorPolicy
  {
   public:
    PoissonRegressionModel(int xdim);
    PoissonRegressionModel(const Vec& beta);
    virtual  PoissonRegressionModel * clone()const;

    virtual GlmCoefs & coef();
    virtual const GlmCoefs & coef()const;
    virtual Ptr<GlmCoefs> coef_prm();
    virtual const Ptr<GlmCoefs> coef_prm()const;

    virtual double Loglike(Vec &g, Mat &h, uint nd)const;
    double log_likelihood(const Vec &beta, Vec *g = NULL, Mat *h = NULL)const;
    virtual double pdf(const Data *, bool logscale)const;
    double logp(const PoissonRegressionData &data)const;
  };

} // namespace BOOM

#endif // POISSON_REGRESSION_MODEL_HPP
