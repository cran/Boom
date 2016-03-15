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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_NONPARAMETRIC_BAYES_RULES_MODEL_HPP_
#define BOOM_NONPARAMETRIC_BAYES_RULES_MODEL_HPP_

#include <cpputil/report_error.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Vector.hpp>
#include <Models/Glm/Glm.hpp>
#include <Models/Nonparametric/BayesRules.hpp>
#include <Models/Nonparametric/DataTableDataPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <stats/DataTable.hpp>
#include <vector>

namespace BOOM {
  // A class for a generalized linear model where the design matrix has
  // a nonparametric prior based on the BayesRules basis expansion.
  class BayesRulesModel
      : public DataTableDataPolicy,
        public PriorPolicy
  {
   public:
    // Args:
    //  data_table: the data table for the model with measured
    //    covariates for the data.
    //  responses: the responses for the data.
    BayesRulesModel(const DataTable & data_table);
    BayesRulesModel(const BayesRulesModel & rhs);
    BayesRulesModel * clone() const override = 0;

    ////////////////////
    // TODO(stevescott): Use an existing set of sufficient statistics
    // for this.  Also, remove it from this class and place it in the
    // sampler.
    // ====== sufficient statistics ========

    // The full design matrix for all rules, regardless of whether or
    // not they are active.
    Matrix full_design_matrix() const;

    // The design matrix for the set of active rules (rules with
    // nonzero coefficients).
    Matrix active_design_matrix(const Selector &inc) const;

    // X^T X for the design matrix.
    // TODO(user): Since xtx() is implemented in the posterior
    // sampler, there may be no need for either xtx or xty. Since X is
    // dependent on the state of the sampler, it is defined therein,
    // and so are xtx and xty.
    SpdMatrix xtx() const;

    // number of observations
    double n() const;

    // The number of coefficients in the linear model given the active
    // rules in the selector.
    int number_of_active_coefficients(const Selector &inc) const;

    // ====== parameters =======

    // The GlmCoefs corresponding to the coefficients.
    virtual const Vector & coef() const = 0;

    // ======= predictors =====

    // Returns the linear prediction based on the coefficients and
    // active rules in the model.
     Vector linear_predictor() const;

    // ======= rules ===========

    // Returns the selector of which rules are included in the model
    // from the bag of rules.
    const Selector & included_rules() const;

    void set_included_rules(const Selector & new_included_rules);

    // Returns the bag of rules in the model.
    const std::vector<Ptr<BaseRule>> & rules() const;

    // Number of possible coefficients based on the rules_.
    int nvars_possible() const;

    // Set the indexed rule which_rule to the pointer new_rule.
    void set_rule(int which_rule, Ptr<BaseRule> new_rule);

    // Add a rule to the list of rules in the model. The
    // included_rules_ selector is updated according to whether or not
    // the new rule is included in the model.
    void add_rule(Ptr<BaseRule> new_rule, bool is_included);

    // Delete the indexed rule from the model and update the
    // included_rules_ selector
    void delete_rule(int which_rule);

    // The number of rules possible in the model. This does NOT
    // include the intercept.
    int nrules_possible() const;

    // The intercept rule.
    const Ptr<InterceptRule> intercept() const;

   private:
    // Selector of which rules are to be included in the model.
    Selector included_rules_;

    Ptr<InterceptRule> intercept_;
    Matrix design_matrix_;
    std::vector<Ptr<BaseRule>> rules_;
    SpdMatrix xtx_;
  };
}  // namespace BOOM

#endif  // BOOM_NONPARAMETRIC_BAYES_RULES_MODEL_HPP_
