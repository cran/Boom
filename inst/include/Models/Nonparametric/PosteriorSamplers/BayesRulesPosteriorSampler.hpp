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

#ifndef BOOM_NONPARAMETRIC_BAYES_RULES_POSTERIOR_SAMPLER_HPP_
#define BOOM_NONPARAMETRIC_BAYES_RULES_POSTERIOR_SAMPLER_HPP_

#include <vector>

#include <Models/Nonparametric/BayesRules.hpp>
#include <Models/Nonparametric/BayesRulesModel.hpp>
#include <Models/Nonparametric/PosteriorSamplers/BagOfRulesPrior.hpp>
#include <Models/Nonparametric/PosteriorSamplers/ProposalDistributions.hpp>

#include <Bmath/Bmath.hpp>
#include <cpputil/math_utils.hpp>
#include <distributions.hpp>
#include <distributions/rng.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/MvnGivenSigma.hpp>
#include <Models/MvnGivenScalarSigma.hpp>
#include <Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM {

  class BayesRulesPosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //  model: the model object to sampled.
    //  expected_model_size: how many variables do we expect to see in
    //    the model a priori.
    //  w: a weight between 0 and 1 which allows us to prevent
    //    numberical instabilities associated with colinearity in
    //    observed covariates. If w=0,
    //               Omega inverse = 'prior_nobs' * XTX/n.
    //    If w=1,
    //               Omega inverse = 'prior_nobs' * diag(XTX)/n.
    //  rule_prior_probabilities: prior probabilities for the four
    //    types of rules to be seen in the bag of rules. The order of
    //    the entries in the vector correspond to linear, split, spline,
    //    and interaction rules.
    //  interaction_rule_proposal_probabilities: proposal probabilities
    //    for the four types of rules when building an interaction rule.
    //  spline_expansion_order: order of the splines to be used in the
    //    bag of rules.
    BayesRulesPosteriorSampler(
        BayesRulesModel * model,
        double expected_model_size,
        std::function<double(int)> log_prior_number_rules_in_bag,
        std::function<double(int)> log_prior_number_rules_in_interaction,
        const MarginalRuleTypeProbabilities &prior_rule_type_probabilities,
        const InteractionRuleProposalProbabilities &proposal_probabilities,
        int spline_expansion_order,
        RNG & seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    //========= posterior draws ===========
    // Performs a posterior draw of all model parameters.
    void draw_model_structure();
    virtual void draw_parameters_given_structure() = 0;
    virtual void draw_latent_data() = 0;

    // Returns the marginal likelihood of the model, conditional on a
    // vector of rule inclusion indicators.
    // Args:
    //   included_rules:  Indicates which rules have nonzero coefficients.
    // Returns:
    //   Log marginal likelihood of the data, conditional on the set
    //   of rules and included_rules, but integrating over other model
    //   parameters.
    virtual double log_model_probability(
        const Selector &included_rules) const = 0;

    //========= posterior draws ===========
    // Samples which rules are to be included in the model by a series
    // of MH steps implemented in mcmc_flip_rule_inclusion_indicator.
    virtual void sample_rule_inclusion_indicators();

    // Modifies all included rules via MH steps.
    virtual void modify_active_rules();

    // Probabilistically takes out or adds an inactive rule in the bag
    // of rules.
    // TODO(user): Check this function.
    // TODO(user): Check this function.
    virtual void create_or_delete_inactive_rules();

    // MCMC step attempting to flip one 'bit' in the inclusion vector
    // that indicates which rules have nonzero coefficients.
    //
    // Args:
    //   included_rules: Indicates which rules currently have nonzero
    //     coefficients.
    //   which_rule:  Index of the rule we're thinking about changing.
    //   logp_old:  Log model probability of the current model.
    //
    // Returns:
    //   log model probability of the model with included_rules set at
    //   its new configuration.  Suitable for feeding into next call
    //   of mcmc_flip_rule_inclusion_indicator.
    virtual double mcmc_flip_rule_inclusion_indicator(
        Selector & included_rules,
        int which_rule,
        double logp_old);

    // Returns false iff the rule has a basis element of all zeroes.
    bool check_rule(const BaseRule &rule) const;

    const std::vector<Ptr<BaseRule>> &bag_of_rules() const;

    const BagOfRulesPrior & bag_of_rules_prior() const {
      return *bag_of_rules_prior_;}
    const VariableSelectionPrior & rule_inclusion_prior() const {
      return *rule_inclusion_prior_; }
    void set_rule_inclusion_prior(VariableSelectionPrior *new_prior) {
      rule_inclusion_prior_.reset(new_prior); }

    double expected_model_size() const {return expected_model_size_;}

    // Draws a rule from prior and adds it to the bag.
    virtual void add_rule_to_bag(const Selector & included_rules);

    // Chooses one of the inactive rules with equal probability, and
    // accepts/rejects this change with MH step.  Calling this
    // function with an empty selector will throw an exception,
    // possibly crashing the program.
    virtual void delete_rule_from_bag(const Selector & included_rules);


    // Attempt a Metropolis Hastings move which attempts to replace a
    // specific rule with another rule from the same family, randomly
    // drawn from the proposal distribution for that type of rule.
    //
    // Args:
    //   included_rules:  Indicates which rules have nonzero coefficients.
    //   which_rule:  The index of the rule we're attempting to modify.
    //   logp_current: The log model probability of the original
    //     unaltered model.
    //
    // Returns:
    //   The log model probability of the new model.
    virtual double mcmc_modify_one_rule(int which_rule, double logp_current);

   private:
    virtual double set_posterior_parameters(
        const Selector &included_rules,
        bool compute_log_determinant) const = 0;

    // The model whose parameters are to be drawn.
    BayesRulesModel * m_;

    // A marginal prior for the set of 0's and 1's indicating which
    // rules are included in the model.
    Ptr<VariableSelectionPrior> rule_inclusion_prior_;

    // The expected number of variables in the model a priori.
    double expected_model_size_;

    // The prior distribution over the structure of the bag of rules.
    Ptr<BagOfRulesPrior> bag_of_rules_prior_;

    // A factory class that knows how to make MH proposal
    // distributions for modifying rules.
    ModifyRuleProposalDistributionFactory factory_;
  };

}  // namespace BOOM

#endif  // BOOM_NONPARAMETRIC_BAYES_RULES_POSTERIOR_SAMPLER_HPP_
