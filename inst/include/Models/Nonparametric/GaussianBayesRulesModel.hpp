 /*
  Copyright (C) 2005-2015 Steven L. Scott

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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA
 */

#ifndef BOOM_NONPARAMETRIC_GAUSSIAN_BAYES_RULES_MODEL_HPP_
#define BOOM_NONPARAMETRIC_GAUSSIAN_BAYES_RULES_MODEL_HPP_

#include <vector>

#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <Models/GaussianModelBase.hpp>
#include <Models/Nonparametric/BayesRules.hpp>
#include <Models/Nonparametric/BayesRulesModel.hpp>
#include <stats/DataTable.hpp>

namespace BOOM {

  // A class for a linear model with normally distributed residuals
  // where the design matrix has a nonparametric prior based on the
  // BayesRules basis expansion.

  class GaussianBayesRulesModel
      : public BayesRulesModel,
        public ParamPolicy_2<VectorParams, UnivParams>
  {
   public:
    //======= constructors =======
    // Args:
    //  data_table: the data table containing the predictors to use in
    //    the model.
    //  responses: the vector of responses to be modeled.
    GaussianBayesRulesModel(const DataTable & predictors,
                            const Vector & responses);
    GaussianBayesRulesModel(const GaussianBayesRulesModel &rhs);
    GaussianBayesRulesModel * clone() const override;

    // The vector of responses to be modeled.
    Vector responses() const {return responses_;}
    double sample_variance() const {
      return marginal_sufficient_statistics_.sample_var();
    }
    double yty() const { return marginal_sufficient_statistics_.sumsq(); }


    // ======= parameters ========
    // The conditional linear model coefficients.
    Ptr<VectorParams> Coef_prm();
    const Vector & coef() const override;
    void set_coef(const Vector & new_coef);

    // The residual variance parameter.
    double sigsq() const;
    Ptr<UnivParams> Sigsq_prm();
    void set_sigsq(double sigsq);

   private:
    Vector responses_;
    GaussianSuf marginal_sufficient_statistics_;
  };

}  // namespace BOOM

#endif  // BOOM_NONPARAMETRIC_GAUSSIAN_BAYES_RULES_MODEL_HPP_
