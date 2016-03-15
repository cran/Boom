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

#ifndef BOOM_BAG_OF_RULES_PRIOR_HPP_
#define BOOM_BAG_OF_RULES_PRIOR_HPP_

#include <Models/Nonparametric/BayesRules.hpp>
#include <Models/ModelTypes.hpp>
#include <functional>
#include <distributions/rng.hpp>

namespace BOOM {

  // A class for turning an IntModel into a std::function<double(int)>.
  class IntModelWrapper {
   public:
    IntModelWrapper(IntModel *model) : model_(model) {}
    IntModelWrapper(Ptr<IntModel> model) : model_(model) {}
    double operator()(int x) const {return model_->logp(x);}
   private:
    Ptr<IntModel> model_;
  };

  class MarginalRuleTypeProbabilities {
   public:
    // Pass by value (rather than reference) is intentional.
    MarginalRuleTypeProbabilities(Vector probabilities);
    double linear() const {return linear_rule_prior_probability_;}
    double log_linear() const {return log_linear_rule_prior_probability_;}
    double split() const {return split_rule_prior_probability_;}
    double log_split() const {return log_split_rule_prior_probability_;}
    double spline() const {return spline_rule_prior_probability_;}
    double log_spline() const {return log_spline_rule_prior_probability_;}
    double interaction() const {return interaction_rule_prior_probability_;}
    double log_interaction() const {
      return log_interaction_rule_prior_probability_;}

   private:
    double linear_rule_prior_probability_;
    double log_linear_rule_prior_probability_;

    double split_rule_prior_probability_;
    double log_split_rule_prior_probability_;

    double spline_rule_prior_probability_;
    double log_spline_rule_prior_probability_;

    double interaction_rule_prior_probability_;
    double log_interaction_rule_prior_probability_;
  };

  //----------------------------------------------------------------------
  // Prior distribution for the bag of rules.  Can evaluate the log
  // prior probability of a given rule, and simulate specific or
  // randomly chosen rules.
  class BagOfRulesPrior : private RefCounted {
    friend void intrusive_ptr_add_ref(BagOfRulesPrior * r) {r->up_count();}
    friend void intrusive_ptr_release(BagOfRulesPrior * r) {
      r->down_count();
      if(r->ref_count()==0) delete r;
    }

   public:
    BagOfRulesPrior(
        const DataTable *data_table,
        std::function<double(int)> log_prior_number_rules_in_bag,
        std::function<double(int)> log_prior_number_rules_in_interaction,
        const MarginalRuleTypeProbabilities &rule_type_probabilities,
        int spline_expansion_order);

    RuleType draw_rule_type(RNG &rng) const;
    Ptr<BaseRule> draw_rule(RNG &rng) const;

    // Sampling specific types of rules
    Ptr<LinearRule> draw_linear_rule(RNG &rng) const;
    Ptr<SplitRule> draw_split_rule(RNG &rng) const;
    Ptr<SplineRule> draw_spline_rule(RNG &rng) const;
    Ptr<InteractionRule> draw_interaction_rule(RNG &rng) const;

    // Evaluate the log of the prior density for a
    double log_prior_bag_of_rules(
        const std::vector<Ptr<BaseRule>> &rules) const;

    // Evaluate the log prior of a single rule.
    double logp(const BaseRule &rule) const;

    double log_prior_linear_rule(const LinearRule & rule) const;
    double log_prior_split_rule(const SplitRule & rule) const;
    double log_prior_spline_rule(const SplineRule & rule) const;
    double log_prior_interaction_rule(const InteractionRule & rule) const;

    int number_of_rule_types() const;
    double log_prior_number_rules_in_bag(int number_rules) const;
    int spline_expansion_order() const;

    int nvars() const;

   private:
    // The data table owned by the model.
    const DataTable *data_table_;
    // Summaries of the columns in the data table.
    std::vector<VariableSummary> variable_summaries_;

    std::function<double(int)> log_prior_number_rules_in_bag_;
    std::function<double(int)> log_prior_number_rules_in_interaction_;

    MarginalRuleTypeProbabilities rule_type_probabilities_;

    int spline_expansion_order_;
  };


}  // namespace BOOM

#endif //  BOOM_BAG_OF_RULES_PRIOR_HPP_
