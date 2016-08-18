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

#include <Models/StateSpace/StateSpacePoissonModel.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp>
#include <stats/moments.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  namespace {
    typedef StateSpacePoissonModel SSPM;
    typedef StateSpace::AugmentedPoissonRegressionData APRD;
  }

  APRD::AugmentedPoissonRegressionData(
      double counts, double exposure, const Vector &predictors)
      : PoissonRegressionData(counts, predictors, exposure),
        latent_continuous_value_(0.0),
        variance_(1.0),
        offset_(0.0)
  {}

  void APRD::set_latent_data(double value, double variance) {
    latent_continuous_value_ = value;
    if (variance < 0) {
      report_error("variance must be positive.");
    }
    variance_ = variance;
  }

  void APRD::set_offset(double offset) {
    offset_ = offset;
  }

  //======================================================================
  SSPM::StateSpacePoissonModel(int xdim)
      : StateSpaceNormalMixture(xdim > 1),
        observation_model_(new PoissonRegressionModel(xdim))
  {}

  SSPM::StateSpacePoissonModel(const Vector &counts,
                               const Vector &exposure,
                               const Matrix &design,
                               const std::vector<bool> &observed)
      : StateSpaceNormalMixture(ncol(design) > 0),
        observation_model_(new PoissonRegressionModel(ncol(design)))
  {
    if ((ncol(design) == 1) &&
        (var(design.col(0)) < std::numeric_limits<double>::epsilon())) {
      set_regression_flag(false);
    }
    bool all_observed = observed.empty();
    if (counts.size() != exposure.size()
        || counts.size() != nrow(design)
        || (!all_observed && counts.size() != observed.size())) {
      report_error("Data sizes do not match in StateSpacePoissonModel "
                   "constructor");
    }
    for (int i = 0; i < counts.size(); ++i) {
      bool missing = !(all_observed || observed[i]);
      NEW(APRD, dp)(missing ? 0 : counts[i],
                    missing ? 0 : exposure[i],
                    design.row(i));
      if (missing) {
        dp->set_missing_status(Data::missing_status::completely_missing);
      }
      add_data(dp);
    }
  }

  SSPM::StateSpacePoissonModel(const SSPM &rhs)
      : StateSpaceNormalMixture(rhs),
        observation_model_(rhs.observation_model_->clone())
  {}

  SSPM * SSPM::clone() const {
    return new SSPM(*this);
  }

  int SSPM::time_dimension() const {
    return dat().size();
  }

  double SSPM::observation_variance(int t) const {
    return dat()[t]->latent_data_variance();
  }

  double SSPM::adjusted_observation(int t) const {
    if (is_missing_observation(t)) {
      return negative_infinity();
    }
    return dat()[t]->latent_data_value()
        - observation_model_->predict(dat()[t]->x());
  }

  bool SSPM::is_missing_observation(int t) const {
    return dat()[t]->missing() != Data::observed;
  }

  void SSPM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      double offset = observation_matrix(t).dot(state(t));
      dat()[t]->set_offset(offset);
      signal_complete_data_change(t);
    }
  }

  Vector SSPM::simulate_forecast(const Matrix &forecast_predictors,
                                 const Vector &exposure,
                                 const Vector &final_state) {
    StateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(nrow(forecast_predictors));
    Vector state = final_state;
    int t0 = dat().size();
    for (int t = 0; t < ans.size(); ++t) {
      state = simulate_next_state(state, t + t0);
      double eta = observation_matrix(t + t0).dot(state)
          + observation_model_->predict(forecast_predictors.row(t));
      double mu = exp(eta);
      ans[t] = rpois(exposure[t] * mu);
    }
    return ans;
  }

  Vector SSPM::one_step_holdout_prediction_errors(
      RNG &rng,
      PoissonDataImputer &data_imputer,
      const Vector &counts,
      const Vector &exposure,
      const Matrix &predictors,
      const Vector &final_state) {
    if (nrow(predictors) != counts.size()
        || exposure.size() != counts.size()) {
      report_error("Size mismatch in arguments provided to "
                   "one_step_holdout_prediction_errors.");
    }
    Vector ans(counts.size());
    int t0 = dat().size();
    ScalarKalmanStorage ks(state_dimension());
    ks.a = *state_transition_matrix(t0 - 1) * final_state;
    ks.P = SpdMatrix(state_variance_matrix(t0 - 1)->dense());

    // There function differs from one_step_holdout_prediction_errors
    // in StateSpaceRegressionModel because the response is on the
    // Poisson scale, and the state needs a non-linear (exp) transform
    // to get it on the scale of the data.  We handle this by imputing
    // the latent data for each observation, using the latent data to
    // sample an observation on which the one-step holdout will be
    // computed, and then updating the Kalman filter to draw the next
    // time point.
    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      // 1) simulate next state.
      // 2) simulate w_t given state
      // 3) kalman update state given w_t.
      Vector state = rmvn(ks.a, ks.P);

      double state_contribution = observation_matrix(t+t0).dot(state);
      double regression_contribution =
          observation_model_->predict(predictors.row(t));
      double mu = state_contribution + regression_contribution;
      double prediction = exposure[t] * exp(mu);
      ans[t] = counts[t] - prediction;

      // ans[t] is a random draw of the one step ahead prediction
      // error at time t0+t given observed data to time t0+t-1.  We
      // now proceed with the steps needed to update the Kalman filter
      // so we can compute ans[t+1].

      double internal_neglog_final_event_time = 0;
      double internal_mixture_mean = 0;
      double internal_mixture_precision = 0;
      double neglog_final_interarrival_time = 0;
      double external_mixture_mean = 0;
      double external_mixture_precision = 0;

      data_imputer.impute(
          rng,
          counts[t],
          exposure[t],
          mu,
          &internal_neglog_final_event_time,
          &internal_mixture_mean,
          &internal_mixture_precision,
          &neglog_final_interarrival_time,
          &external_mixture_mean,
          &external_mixture_precision);

      double total_precision = external_mixture_precision;
      double precision_weighted_sum =
          neglog_final_interarrival_time - external_mixture_mean;
      precision_weighted_sum *= external_mixture_precision;
      if (counts[t] > 0) {
        precision_weighted_sum +=
            (internal_neglog_final_event_time - internal_mixture_mean)
            * internal_mixture_precision;
        total_precision += internal_mixture_precision;
      }
      double latent_observation = precision_weighted_sum / total_precision;
      double latent_variance = 1.0 / total_precision;

      // The latent state was drawn from its predictive distribution
      // given Y[t0 + t -1] and used to impute the latent data for
      // y[t0+t].  That latent data is now used to update the Kalman
      // filter for the next time period.  It is important that we
      // discard the imputed state at this point.
      sparse_scalar_kalman_update(latent_observation - regression_contribution,
                                  ks.a,
                                  ks.P,
                                  ks.K,
                                  ks.F,
                                  ks.v,
                                  missing,
                                  observation_matrix(t + t0),
                                  latent_variance,
                                  *state_transition_matrix(t + t0),
                                  *state_variance_matrix(t + t0));
    }
    return ans;
  }

}  // namespace BOOM
