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

#ifndef BOOM_NONPARAMETRIC_BAYES_RULES_PROPOSAL_DISTRIBUTIONS_HPP_
#define BOOM_NONPARAMETRIC_BAYES_RULES_PROPOSAL_DISTRIBUTIONS_HPP_

#include <Models/Nonparametric/BayesRules.hpp>
#include <Models/Nonparametric/BayesRulesModel.hpp>

// This fule contains proposal distributions for the Metroplis
// Hastings algorithm that samples which rules are in the "bag of
// rules" that are candidates for basis expansions.  The basic
// structure for the MH algorithm is to consider three high level
// moves:
//
// I) Add an inactive rule to the bag.
// II) Delete an inactive rule from the set of inactive rules.
// III) Modify one of the active rules.
//
// The classes in this file focus on III.
namespace BOOM {

  //======================================================================
  // Base class for "modify rule" proposal distributions.
  //
  class ModifyRuleProposalBase : private RefCounted {
    friend void intrusive_ptr_add_ref(ModifyRuleProposalBase * r) {
      r->up_count();}
    friend void intrusive_ptr_release(ModifyRuleProposalBase * r) {
      r->down_count();
      if (r->ref_count()==0) { delete r; }
    }

   public:
    // Args:
    //   data_table: Points to the data table for the BayesRules
    //     model.  An invalid or NULL pointer will cause an exception.
    ModifyRuleProposalBase(const DataTable *data_table);
    ~ModifyRuleProposalBase() override {}

    // Evaluate the log density of a move from the current rule to the
    // proposed rule.
    virtual double logp() const = 0;

    // Evaluate the log density of a move from the proposed rule to
    // the current_rule.
    virtual double reverse_logp() const = 0;

    // Propose a new rule from the appropriate family.  This method is
    // non-const because the proposed rule will be stored internally.
    virtual BaseRule * propose_rule(RNG &rng) = 0;

    const DataTable &data_table() const {
      return *data_table_;
    }

   private:
    const DataTable *data_table_;
  };

  //======================================================================
  // Class for proposing linear rules and evaluating their proposal
  // densities.
  class LinearRuleProposal : public ModifyRuleProposalBase {
   public:
    LinearRuleProposal(Ptr<LinearRule> current_rule,
                       const DataTable *data_table);
    LinearRule * propose_rule(RNG &rng) override;
    double logp() const override;
    double reverse_logp() const override;
   private:
    Ptr<LinearRule> current_rule_;
    Ptr<LinearRule> proposed_rule_;
  };

  //=====================================================================
  // Object for proposing split POINTS and evaluating their proposal
  // densities.
  class SplitPointProposal  {
   public:
    SplitPointProposal(double current_split_point,
                       double min,
                       double max,
                       double epsilon);

    // Draws a point from the interval [min, max] uniformly.
    double propose_split_point_uniform(RNG &rng);

    // Draws a point uniformly from a neighborhood of size epsilon
    // around the current split point.
    double propose_split_point_uniform_neighborhood(RNG &rng);

    double logp() const;
    double reverse_logp() const;
    double proposed_point() const;
    double current_point() const;

   private:
    double current_split_point_;
    double min_;
    double max_;
    double epsilon_;

    double proposed_split_point_;
  };

  //=====================================================================
  // Object for proposing split rules and evaluating their proposal
  // densities.
  class SplitRuleProposal : public ModifyRuleProposalBase {
   public:
    // Args:
    //   current_rule:  The current rule from which to make proposals.
    //   data_table:  THe data table used to fit the model.
    //   min: The smallest observed value for the X variable being
    //     split by the current rule.
    //   max:  The largest observed value for the X variable being
    //     split by the current rule.
    //   epsilon: The size of the neighborhood around the current
    //     split point in which to look for better split points.
    SplitRuleProposal(Ptr<SplitRule> current_rule,
                      const DataTable *data_table,
                      double min,
                      double max,
                      double epsilon);
    SplitRule * propose_rule(RNG &rng) override;
    double logp() const override;
    double reverse_logp() const override;

   private:
    Ptr<SplitRule> current_rule_;
    SplitPointProposal split_point_proposal_;
    Ptr<SplitRule> proposed_rule_;
  };

  //=====================================================================
  // Object for proposing spline rules and evaluating their proposal
  // densities.
  class SplineRuleProposal : public ModifyRuleProposalBase {
   public:
    // Args:
    //   current_rule:  The current rule from which to make proposals;
    //   data_table:  The data table used to fit the model.
    //   spline_order:  The degree of the smoothing spline.
    //   xmin: The smallest obseved x value for the variable in the
    //     current rule.
    //   xmax: The largest obseved x value for the variable in the
    //     current rule.
    SplineRuleProposal(Ptr<SplineRule> current_rule,
                       const DataTable *data_table,
                       int spline_order,
                       double xmin,
                       double max);
    double logp() const override;
    double reverse_logp() const override;
    SplineRule * propose_rule(RNG &rng) override;

   private:
    Ptr<SplineRule> current_rule_;
    int order_;
    double xmin_;
    double xmax_;
    Ptr<SplineRule> proposed_rule_;
  };

  //=====================================================================
  // Object for proposing interaction rules and evaluating the proposal
  // density for proposed rule given the current rule in MH steps.
  // Something to keep track of the proposal probabilities
  class InteractionRuleProposalProbabilities {
   public:
    // Args:
    //   probabilities: A 3-element vector with elements corresponding
    //     to "modify", "delete", and "add" moves.  Pass by value
    //     (rather than reference) is deliberate here.
    InteractionRuleProposalProbabilities(Vector probabilities);
    double modification() const {
      return probability_of_proposing_a_modification_;}
    double deletion() const {
      return probability_of_proposing_a_deletion_;}
    double addition() const {
      return probability_of_proposing_an_addition_;}

   private:
    double probability_of_proposing_a_modification_;
    double probability_of_proposing_a_deletion_;
    double probability_of_proposing_an_addition_;
  };

  class ModifyRuleProposalDistributionFactory;

  class InteractionRuleProposal : public ModifyRuleProposalBase {
   public:
    InteractionRuleProposal(Ptr<InteractionRule> current_rule,
                            const BayesRulesModel *model,
                            const InteractionRuleProposalProbabilities &
                            proposal_probabilities,
                            ModifyRuleProposalDistributionFactory *factory,
                            const BagOfRulesPrior *prior);
    enum MoveType {
      ERROR = -1,
      MODIFY_RULE,
      DELETE_RULE,
      ADD_RULE
    };

    double logp() const override {return logp_;}
    double reverse_logp() const override {return reverse_logp_;}
    InteractionRule *propose_rule(RNG &rng) override;

    InteractionRule *propose_modifying_rule(RNG &rng);
    InteractionRule *propose_adding_rule(RNG &rng);
    InteractionRule *propose_deleting_rule(RNG &rng);

    const BagOfRulesPrior * bag_of_rules_prior() const {return prior_;}

   private:
    Ptr<InteractionRule> current_rule_;
    const BayesRulesModel *model_;
    InteractionRuleProposalProbabilities proposal_probabilities_;
    Ptr<InteractionRule> proposed_rule_;
    double logp_;
    double reverse_logp_;
    ModifyRuleProposalDistributionFactory *factory_;
    const BagOfRulesPrior *prior_;

    MoveType select_move_type(RNG &rng) const;
  };

  //======================================================================
  // A factory class for creating proposal distributions that modify
  // existing rules.
  class ModifyRuleProposalDistributionFactory {
   public:
    ModifyRuleProposalDistributionFactory(
        const BayesRulesModel *model,
        const InteractionRuleProposalProbabilities
        &interaction_rule_proposal_probabilities,
        int spline_order,
        const BagOfRulesPrior *prior);

    LinearRuleProposal *create_linear_rule_proposal(LinearRule *rule);
    SplitRuleProposal *create_split_rule_proposal(SplitRule *rule);
    SplineRuleProposal *create_spline_rule_proposal(SplineRule *rule);
    InteractionRuleProposal *create_interaction_rule_proposal(
        InteractionRule *rule);

   private:
    const BayesRulesModel *model_;
    std::vector<VariableSummary> summaries_;
    InteractionRuleProposalProbabilities interaction_proposal_probabilities_;
    int spline_order_;
    const BagOfRulesPrior *prior_;
  };


} // namespace BOOM

#endif //  BOOM_NONPARAMETRIC_BAYES_RULES_PROPOSAL_DISTRIBUTIONS_HPP_
