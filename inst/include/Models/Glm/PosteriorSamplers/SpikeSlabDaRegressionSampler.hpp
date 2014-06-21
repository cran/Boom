/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_SPIKE_SLAB_DA_REGRESSION_SAMPLER_HPP_
#define BOOM_SPIKE_SLAB_DA_REGRESSION_SAMPLER_HPP_

#include <Models/Glm/RegressionModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/IndependentMvnModelGivenScalarSigma.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/MvnGivenSigma.hpp>
#include <Models/GammaModel.hpp>

namespace BOOM {

  // A posterior sampler for linear models under a spike and slab
  // prior based on the data augmentation algorithm from Clyde and
  // Ghosh (2011).
  class SpikeSlabDaRegressionSampler : public PosteriorSampler {
   public:
    // Args:
    //   model: The model for which posterior draws are desired.  The
    //     data should have already been assigned to the model before
    //     being passed here.  The sampler will use information about
    //     the design matrix that cannot be changed after the
    //     constructor is called.  The first column of the design
    //     matrix for the model is assumed to contain an intercept
    //     (all 1's).
    //   beta_prior: Prior distribution for regression coefficients,
    //     given inclusion.
    //   siginv_prior:  Prior distribution for the residual variance.
    //   prior_inclusion_probabilities: Prior probability that each
    //     variable is "in" the model.
    SpikeSlabDaRegressionSampler(
        RegressionModel *model,
        Ptr<IndependentMvnModelGivenScalarSigma> beta_prior,
        Ptr<GammaModelBase> siginv_prior,
        const Vector & prior_inclusion_probabilities);

    virtual double logpri()const;
    virtual void draw();

    // Compute the inclusion probability of coefficient i given complete
    // data.  The complete data makes all the coefficients independent.
    double compute_inclusion_probability(int i)const;

    double prior_ss()const;
    double prior_df()const;

    // The prior information for variable j.
    double unscaled_prior_information(int j)const;
    double prior_information(int j)const;

    //----------------------------------------------------------------------
    // Views of private objects, exposed for testing.
    const Vector &log_prior_inclusion_probabilities() const {
      return log_prior_inclusion_probabilities_; }
    const Vector &log_prior_exclusion_probabilities() const {
      return log_prior_exclusion_probabilities_; }
    const Matrix &missing_design_matrix()const{return missing_design_matrix_;}
    const Vector &complete_data_xtx_diagonal() const {
      return complete_data_xtx_diagonal_;}
    const Vector &missing_y() const {return missing_y_;}
    const Vector &complete_data_xty()const{return complete_data_xty_;}
    const double &complete_data_yty()const{return complete_data_yty_;}

   private:
    // NOTE: This function assumes that the original X matrix had a
    // column of 1's in the first column.
    //
    // 1) Scale the observed design matrix by dividing each column by
    //    its standard deviation.  Columns with zero standard
    //    deviation are left unchanged.  Record the standard
    //    deviations in a "diagonal matrix" S.
    // 2) Set D = diag(largest eigenvalue of the scaled matrix).
    // 3) Find W = chol(D - XTX).t()
    // 4) Rescale W = W * S, and D = S * D * S.
    void determine_missing_design_matrix();
    void impute_latent_data();

    // The draw of the model indicators is given sigma, but with beta
    // integrated out.  Otherwise the relevant sums of squares do not
    // factor as a product of per-variable contributions.
    void draw_model_indicators();

    // Note, the draw of beta is given sigma.  Integrating over sigma
    // (i.e. not conditioning on it) would make the marginal
    // distribution of the model indicators polynomial (e.g. it would
    // look like the T distribution) instead of exponential
    // (e.g. looking like the Gaussian distribution).  We need it to
    // be exponential so it factors variable-by-variable.
    void draw_beta_given_complete_data();

    void draw_sigma_given_complete_data();

    void observe_changes_in_prior()const;
    void check_prior()const;

    double information_weighted_prior_mean(int j)const;
    double posterior_mean_beta_given_complete_data(int j)const;

    RegressionModel *model_;
    Ptr<IndependentMvnModelGivenScalarSigma> beta_prior_;
    Ptr<GammaModelBase> siginv_prior_;

    // The prior probability that each varaiable is included in the
    // model.  The intercept can be excluded just as any other
    // variable.
    Vector log_prior_inclusion_probabilities_;

    // exp(log_prior_exclusion_probabilities_) =
    // 1-exp(log_prior_inclusion_probabilities_)
    Vector log_prior_exclusion_probabilities_;

    // The missing design matrix is the upper cholesky triangle of the
    // sum of squares matrix required to diagonalize the cross product
    // matrix of the observed data.
    Matrix missing_design_matrix_;

    // The elements of the response vector corresponding to the rows
    // in missing_design_matrix_.
    Vector missing_y_;

    // The diagonal elements of the posterior information matrix xtx +
    // xtx_missing + prior_information.  The off-diagonal elements are
    // zero by design.  This matrix will be constant once determined
    // by determine_missing_design_matrix().
    //
    // This matrix must be divided by model_->sigsq() to get the
    // posterior Fisher information.
    Vector complete_data_xtx_diagonal_;

    // The elements of the un-normalized posterior mean: xty +
    // xty_missing + prior_information * prior_mean.  The xty_missing
    // portion of this sum will change with each MCMC iteration.
    Vector complete_data_xty_;
    double complete_data_yty_;

    // Prior distribution

    // The prior precision, unscaled by sigma.  This is the prior
    // precision that would be obtained if sigma == 1.
    mutable Vector unscaled_prior_precision_;

    // unscaled_prior_precision_ * prior_mean
    mutable Vector information_weighted_prior_mean_;

    // A flag indicating that the values of the prior distribution for
    // beta are current.  This flag can only be set to false by a call
    // to observe_changes_in_prior(), which is to be set as an
    // observer in the parameters of the prior distribution.
    mutable bool prior_is_current_;
  };
}


#endif //  BOOM_SPIKE_SLAB_DA_REGRESSION_SAMPLER_HPP_
