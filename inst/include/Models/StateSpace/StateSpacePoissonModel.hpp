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

#ifndef BOOM_STATE_SPACE_POISSON_MODEL_HPP_
#define BOOM_STATE_SPACE_POISSON_MODEL_HPP_

#include <Models/StateSpace/StateSpaceNormalMixture.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Glm/PoissonRegressionModel.hpp>
#include <Models/Glm/PoissonRegressionData.hpp>

namespace BOOM {

  class PoissonDataImputer;
  namespace StateSpace {

    // Let y_t denote the non-negative integer observed at time t, and
    // let E_t be a known positive real number (the "exposure").  The
    // observation equation is
    //
    //            y_t ~ Poisson(E_t * lambda_t), where
    //  log(lambda_t) = Z_t^T \alpha_t + \beta * x_t
    //                = mu_t.
    //
    // One can view y_t as the number of events produced by a Poisson
    // process, with rate lambda_t, in an interval of width E_t.  The
    // inter-event times between events is exponential with rate
    // lambda_t.  The sum of the event times (which is the time to the
    // final event, event y_t) is Gamma(y_t, lambda_t).  Let this
    // final event time be tau_t.  Note that lambda_t is just a scale
    // factor in this expression, so tau_t ~ Ga(y_t, 1) / lambda_t.
    // Taking the negative log of this expression yields -log tau_t =
    // mu_t + epsilon_t, where epsilon_t ~ -log Ga(y_t, 1).  The error
    // distribution can be represented as a mixture of normals, with
    // mean and variance depending on y_t.  Note that if y_t == 0 no
    // event occurs in the interval.  Also note that the negative log
    // gamma distribution is not symmetric for small values of y_t, so
    // the normal mixture is over both means and variances.
    //
    // There is also information in E_t - tau_t, the amount of time
    // between the final observation and the end of the exposure
    // interval.  This is captured by simulating the time of the first
    // event outside the interval, delta_t, where delta_t - tau_t ~
    // Exponential(lambda_t) = Exponential(1) / lambda_t.  As above
    // -log(delta_t - tau_t) = mu_t + (error) where (error) follows a
    // mixture of normals.
    //
    // Thus y_t is associated with either one (if y_t == 0) or two
    // (y_t > 0) latent variables which (after subtracting off the
    // mean from the normal mixture), both have mean mu_t and
    // variances v1 and v2.  The information content in these two
    // observations is equivalent to a single Gaussian with mean mu_t
    // and precision 1/v1 + 1/v2.  If y_t == 0, so only a single
    // observation is available then the precision is just 1/v1.
    class AugmentedPoissonRegressionData
        : public PoissonRegressionData {
     public:
      AugmentedPoissonRegressionData(
          double count, double exposure, const Vector &x);

      void set_latent_data(double value, double variance);
      double latent_data_variance() const {return variance_;}
      double latent_data_value() const {return latent_continuous_value_;}
      void set_offset(double offset);
      double offset() const {return offset_;}

     private:
      // If y() > 0 this is (-log(tau_t) - m1)/v1 + (-log(delta_t -
      // tau_t) - m2)/v2/(1/v1 + 1/v2), where m1,v1 and m2,v2 are the
      // normal mixture means and variances.  If y() == 0 then this is
      // just -log(delta_t) - m1.
      double latent_continuous_value_;

      // If y() > 0 this is 1/(1/v1 + 1/v2).  Otherwise it is simply
      // v1.
      double variance_;

      // The offset stores the state contribution to
      // latent_continuous_value_.
      double offset_;
    };

  }  // namespace StateSpace

  class StateSpacePoissonModel
      : public StateSpaceNormalMixture,
        public IID_DataPolicy<StateSpace::AugmentedPoissonRegressionData>,
        public PriorPolicy
  {
   public:
    StateSpacePoissonModel(int xdim);
    StateSpacePoissonModel(const Vector &counts,
                           const Vector &exposure,
                           const Matrix &design_matrix,
                           const std::vector<bool> &observed =
                           std::vector<bool>());

    StateSpacePoissonModel(const StateSpacePoissonModel &rhs);
    StateSpacePoissonModel * clone() const override;

    const StateSpace::AugmentedPoissonRegressionData &
    data(int t) const override {
      return *(dat()[t]);
    }
    int time_dimension() const override;

    // Returns the imputed observation variance from the latent
    // data for observation t.  This is V_t from the class comment
    // above.
    double observation_variance(int t) const override;

    // Returns the imputed value for observation t minus x[t]*beta.
    // Returns -infinity if observation t is missing.
    double adjusted_observation(int t) const override;

    // Returns true if observation t is missing, false otherwise.
    bool is_missing_observation(int t) const override;

    PoissonRegressionModel *observation_model() override {
      return observation_model_.get(); }
    const PoissonRegressionModel *observation_model() const override {
      return observation_model_.get(); }

    // Set the offset in the data to the state contribution.
    void observe_data_given_state(int t) override;

    Vector simulate_forecast(const Matrix &forecast_predictors,
                             const Vector &exposure,
                             const Vector &final_state);

    Vector one_step_holdout_prediction_errors(
        RNG &rng,
        PoissonDataImputer &data_imputer,
        const Vector &response,
        const Vector &exposure,
        const Matrix &predictors,
        const Vector &final_state);

   private:
    Ptr<PoissonRegressionModel> observation_model_;
  };

}  // namespace BOOM

#endif // BOOM_STATE_SPACE_POISSON_MODEL_HPP_
