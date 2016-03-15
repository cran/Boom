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

#ifndef BOOM_NONPARAMETRIC_GAUSSIAN_BAYES_RULES_POSTERIOR_SAMPLER_HPP_
#define BOOM_NONPARAMETRIC_GAUSSIAN_BAYES_RULES_POSTERIOR_SAMPLER_HPP_

#include <cpputil/math_utils.hpp>
#include <cpputil/seq.hpp>
#include <Models/Nonparametric/PosteriorSamplers/BayesRulesPosteriorSampler.hpp>
#include <Models/Nonparametric/GaussianBayesRulesModel.hpp>
#include <Models/ChisqModel.hpp>
#include <vector>

namespace BOOM {
  class GaussianBayesRulesPosteriorSampler
      : public BayesRulesPosteriorSampler {
   public:
    // Omega inverse is 'prior_nobs' * XTX/n. The intercept term in
    // 'b' is ybar (sample mean of the responses).  The slope terms in
    // b are all zero.  The prior for 1/sigsq is Gamma(prior_nobs/2,
    // prior_ss/2), with prior_ss = prior_nobs*sigma_guess^2, and
    // sigma_guess = sample_variance*(1-expected_rsq).  w is a double
    // between 0 and 1 used to deal with possible colinearity in the
    // design matrix.
    // Args:
    //  model: the model object to sampled.
    //  prior_nobs: the information contained in our prior knowledge
    //    measured in number of observations.
    //  expected_rsq: prior belief of how well the model fits the data
    //    in terms of R^2.
    //  expected_model_size: how many variables do we expect to see in
    //    the model a priori.
    //  w: a weight between 0 and 1 which allows us to prevent
    //    numberical instabilities associated with colinearity in
    //    observed covariates. If w=0,
    //             Omega inverse = 'prior_nobs' * XTX/n.
    //    If w=1,
    //             Omega inverse = 'prior_nobs' * diag(XTX)/n.
    //  rule_prior_probabilities: prior probabilities for the four
    //    types of rules to be seen in the bag of rules.
    //  interaction_rule_proposal_probabilities: proposal probabilities
    //    for the four types of rules when building an interaction rule.
    //  spline_expansion_order: order of the splines to be used in the
    //    bag of rules.
    GaussianBayesRulesPosteriorSampler(
        GaussianBayesRulesModel * model,
        double prior_nobs,
        double expected_rsq,
        double expected_model_size,
        double w,
        std::function<double(int)> log_prior_number_rules_in_bag,
        std::function<double(int)> log_prior_number_rules_in_interaction,
        const MarginalRuleTypeProbabilities &rule_type_prior_probabilities,
        const InteractionRuleProposalProbabilities &proposal_probabilities,
        int spline_expansion_order);

    // X^T y for the input and the responses associated with model m_.
    Vector xty(const Matrix & X) const;

    // TODO(user): Belongs in GaussianBayesRulesPosteriorSampler.
    // ========= prior parameters ========
    // Prior degrees of freedom.
    double prior_df() const;
    // Prior ss.
    double prior_ss() const;

    //========= posterior draws ===========
    // Performs a posterior draw of all model parameters.
    void draw_parameters_given_structure() override;

    // Default assumption is that there is no latent data in the model.
    void draw_latent_data() override;


    // Posterior log-probability for the current model.
    double log_model_probability(const Selector &g) const override;

    // ======== re-setting posterior parameters =========
    // Re-evaluating beta_tilde_ and iV_tilde_ after a draw of the
    // rules.
    double set_posterior_parameters(
        const Selector & included_rules,
        bool do_log_determinant_Ominv) const override;

    // Samples posterior linear model coefficients for the rules in
    // the model.
    virtual void sample_coefficients();

    // Samples the posterior residual variance.
    virtual void sample_sigma_sq();

   private:
    // The model to be sampled.
    GaussianBayesRulesModel * m_;

    // A marginal prior distribution for 1/sigma^2.
    Ptr<GammaModelBase> siginv_prior_;

    // A weight between 0 and 1 used to prevent colinearity in Ominv.
    double colinearity_weight_;

    // Posterior mean for the model coefficients.
    // TODO(user): Revisit these members.
    // TODO(user): Revisit these members.
    mutable Vector beta_tilde_;      // this is work space for computing
    mutable SpdMatrix iV_tilde_;        // posterior model probs
    mutable double DF_, SS_;
  };

}  // namespace BOOM

#endif  // BOOM_NONPARAMETRIC_GAUSSIAN_BAYES_RULES_POSTERIOR_SAMPLER_HPP_
