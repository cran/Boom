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

#ifndef BOOM_STATE_SPACE_LOGIT_MODEL_HPP_
#define BOOM_STATE_SPACE_LOGIT_MODEL_HPP_

#include <Models/StateSpace/StateSpaceNormalMixture.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Glm/BinomialLogitModel.hpp>
#include <Models/Glm/BinomialRegressionData.hpp>

namespace BOOM {

  class BinomialLogitDataImputer;

  namespace StateSpace {

    // BinomialRegressionData, augmented with a pair of latent
    // variables.
    class AugmentedBinomialRegressionData
        : public BinomialRegressionData {
     public:
      AugmentedBinomialRegressionData(
          double y, double n, const Vector &x);
      void set_latent_data(double value, double variance);
      double latent_data_variance() const {return variance_;}
      double latent_data_value() const {return latent_continuous_value_;}
      void set_offset(double offset);
      double offset() const {return offset_;}

     private:
      // The precision weighted mean of the underlying Gaussian
      // observations.
      double latent_continuous_value_;

      // The inverse of the sum of the precisions of the underlying
      // latent Gaussians.
      double variance_;

      // The offset stores the state contribution to
      // latent_continuous_value_.
      double offset_;
    };

  }  // namespace StateSpace

  // Let y_t denote the number of successes out of n_t trials, where
  // n_t is taken as a known constant.  The observation equation is
  //
  //         y_t ~ Binomial(n_t, p_t), where
  //  logit(p_t) = Z_t^T \alpha_t + \beta * x_t
  //             = mu_t.
  //
  // Observation y_t is associated with a pair of vectors of random
  // variables z_t, v_t, each of dimension n_t, where z_{it} \sim
  // N(mu_t, v_{it}), and y_t = sum_i I(z_{it} > 0).
  //
  // The weighted average of z_{it}
  //
  //                sum_i z_{it} / v_{it}
  //      zbar_t =  ------------------
  //                 sum_i  1 / v_{it}
  //
  // is a complete data sufficient statistic observation t.  It is Z_t
  // that gets imputed, along with its variance
  //
  //       V_t = 1.0 / sum_i(1.0 / v_{it}).
  //
  class StateSpaceLogitModel
      : public StateSpaceNormalMixture,
        public IID_DataPolicy<StateSpace::AugmentedBinomialRegressionData>,
        public PriorPolicy
  {
   public:
    StateSpaceLogitModel(int xdim);
    StateSpaceLogitModel(const Vector &successes,
                         const Vector &trials,
                         const Matrix &design_matrix,
                         const std::vector<bool> &observed =
                         std::vector<bool>());

    StateSpaceLogitModel(const StateSpaceLogitModel &rhs);
    StateSpaceLogitModel * clone() const override;

    const StateSpace::AugmentedBinomialRegressionData &
    data(int t) const override {
      return *(dat()[t]);
    }

    int time_dimension() const override;

    // Returns the imputed observation variance from the latent
    // data for observation t.  This is V_t from the class comment
    // above.
    double observation_variance(int t) const override;

    // Returns the imputed value for observation t (zbar_t in the
    // the class comment, above), minus x[t]*beta.  Returns
    // infinity if observation t is missing.
    double adjusted_observation(int t) const override;

    // Returns true if observation t is missing, false otherwise.
    bool is_missing_observation(int t) const override;

    BinomialLogitModel *observation_model() override {
      return observation_model_.get(); }
    const BinomialLogitModel *observation_model() const override {
      return observation_model_.get(); }

    // Set the offset in the data to the state contribution.
    void observe_data_given_state(int t) override;

    // Returns a vector of draws from the posterior predictive
    // distribution of the next nrow(forecast_predictors) time
    // periods.  The draws are on the same (binomial) scale as the
    // original data (as opposed to the logit scale).
    //
    // Args:
    //   forecast_predictors: A matrix of predictors to use for the
    //     forecast period.  If no regression component is desired,
    //     then a single column matrix of 1's (an intercept) should be
    //     supplied so that the length of the forecast period can be
    //     determined.
    //   trials: A vector of non-negative integers giving the number
    //     of trials that will take place at each point in the
    //     forecast period.
    //   final_state: A draw of the value of the state vector at the
    //     final time period in the training data.
    Vector simulate_forecast(const Matrix &forecast_predictors,
                             const Vector &trials,
                             const Vector &final_state);

    // Args:
    //   rng:  A U(0,1) random number generator.
    //   data_imputer: A data imputer that can be used to unmix the
    //     binomial observations into a latent logistic, and then to a
    //     mixture of normals.
    //   successes: The vector of success counts during the holdout
    //     period.
    //   trials: The vector of trial counts during the holdout period.
    //   predictors: The matrix of predictors for the holdout period.
    //     If the model contains no regression component then a single
    //     column matrix of 1's should be supplied.
    //   final_state: A draw of the value of the state vector at the
    //     final time period in the training data.
    //
    // Returns:
    //   A draw from the posterior distribution of the one-state
    //   holdout errors.  The draw is on the scale of the original
    //   data, so it will consist of integers, but it is an error, so
    //   it may be positive or negative.
    //
    //  TODO(stevescott): consider whether this would make more sense
    //  on the logit scale.
    Vector one_step_holdout_prediction_errors(
        RNG &rng,
        BinomialLogitDataImputer &data_imputer,
        const Vector &successes,
        const Vector &trials,
        const Matrix &predictors,
        const Vector &final_state);

   private:
    // Registers the observation model with the ParamPolicy, and sets
    // observers on the model parameters to invalidate the Kalman
    // filter if the parameters change values.
    void setup();

    Ptr<BinomialLogitModel> observation_model_;
  };

}  // namespace BOOM

#endif // BOOM_STATE_SPACE_LOGIT_MODEL_HPP_
