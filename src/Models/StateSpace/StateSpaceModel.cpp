/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#include <Models/StateSpace/StateSpaceModel.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>
#include <stats/moments.hpp>
#include <distributions.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  typedef StateSpaceModel SSM;

  void SSM::setup() {
    observe(observation_model_->Sigsq_prm());
    observation_model_->only_keep_sufstats();
    ParamPolicy::add_model(observation_model_);
  }

  SSM::StateSpaceModel()
      : observation_model_(new ZeroMeanGaussianModel)
  {
    setup();
  }

  SSM::StateSpaceModel(const Vector &y, const std::vector<bool> &y_is_observed)
      : observation_model_(new ZeroMeanGaussianModel(
            sqrt(var(y, y_is_observed))/10))
  {
    setup();
    for(int i = 0; i < y.size(); ++i) {
      NEW(DoubleData, dp)(y[i]);
      if(!y_is_observed.empty() && !y_is_observed[i]) {
        dp->set_missing_status(Data::completely_missing);
      }
      add_data(dp);
    }
  }

  SSM::StateSpaceModel(const SSM &rhs)
      : Model(rhs),
        StateSpaceModelBase(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        observation_model_(rhs.observation_model_->clone())
  {
    setup();
  }

  SSM* SSM::clone() const {return new SSM(*this);}

  int SSM::time_dimension() const {return dat().size();}

  double SSM::observation_variance(int) const {
    return observation_model_->sigsq();}

  double SSM::adjusted_observation(int t) const {
    return dat()[t]->value(); }

  bool SSM::is_missing_observation(int t) const {
    return dat()[t]->missing() != Data::observed;
  }

  ZeroMeanGaussianModel* SSM::observation_model() {
    return observation_model_.get();
  }

  const ZeroMeanGaussianModel* SSM::observation_model() const {
    return observation_model_.get();
  }

  void SSM::observe_data_given_state(int t) {
    // Assuming ignorable missing data.
    if(!is_missing_observation(t)) {
      double mu = observation_matrix(t).dot(state(t));
      double y = adjusted_observation(t) - mu;
      observation_model_->suf()->update_raw(y);
    }
  }

  // TODO(stevescott): should observation_matrix and
  // observation_variance be called with t + t0 + 1?
  Matrix SSM::forecast(int n) {
    // TODO(stevescott): This method only works with truly Gaussian
    // state models.  We should put in a check to make sure that none
    // of the state models are T, normal mixture, etc.
    ScalarKalmanStorage ks(filter());
    Matrix ans(n, 2);
    int t0 = time_dimension();
    for(int t = 0; t < n; ++t) {
      ans(t,0) = observation_matrix(t + t0).dot(ks.a);
      sparse_scalar_kalman_update(
          0,    // y is missing, so fill in a dummy value
          ks.a,
          ks.P,
          ks.K,
          ks.F,
          ks.v,
          true,  // forecasts are missing data
          observation_matrix(t + t0),
          observation_variance(t + t0),
          *state_transition_matrix(t + t0),
          *state_variance_matrix(t + t0));
      ans(t,1) = sqrt(ks.F);
    }
    return ans;
  }

  Vector SSM::simulate_forecast(int n, const Vector &final_state) {
    StateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(n);
    int t0 = time_dimension();
    Vector state = final_state;
    for(int t = 0; t < n; ++t) {
      state = simulate_next_state(state, t + t0);
      ans[t] = rnorm(observation_matrix(t + t0).dot(state),
                     sqrt(observation_variance(t + t0)));
    }
    return ans;
  }

  Vector SSM::simulate_forecast_given_observed_data(
      int n, const Vector &observed_data) {
    StateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(n);
    int t0 = observed_data.size();
    ScalarKalmanStorage ks(filter_observed_data(observed_data));
    //  'state' starts out as state[t0], which is one after the final
    //  state.
    Vector state(rmvn(ks.a, ks.P));
    for (int t = 0; t < n; ++t) {
      ans[t] = rnorm(observation_matrix(t + t0).dot(state),
                     sqrt(observation_variance(t + t0)));
      state = simulate_next_state(state, t + t0 + 1);
    }
    StateSpaceModelBase::set_state_model_behavior(StateModel::MIXTURE);
    return ans;
  }

  ScalarKalmanStorage SSM::filter_observed_data(const Vector &observed_data,
                                                int t0) const {
    ScalarKalmanStorage ks(state_dimension());
    ks.a = initial_state_mean();
    ks.P = initial_state_variance();
    ks.F = observation_matrix(0).sandwich(ks.P) + observation_variance(0);
    int n = observed_data.size();
    if(n==0) return ks;

    for(int t = 0; t < n; ++t) {
      double y = observed_data[t];
      bool missing(y == BOOM::negative_infinity());
      sparse_scalar_kalman_update(y,
                                  ks.a,
                                  ks.P,
                                  ks.K,
                                  ks.F,
                                  ks.v,
                                  missing,
                                  observation_matrix(t0 + t),
                                  observation_variance(t0 + t),
                                  *state_transition_matrix(t0 + t),
                                  *state_variance_matrix(t0 + t));
    }
    return ks;
  }

  Vector SSM::one_step_holdout_prediction_errors(
      const Vector &newY,
      const Vector &final_state) const {
    Vector ans(length(newY));
    const std::vector<Ptr<DoubleData> > &data(dat());
    int t0 = data.size();
    ScalarKalmanStorage ks(state_dimension());
    ks.a = *state_transition_matrix(t0 - 1) * final_state;
    ks.P = SpdMatrix(state_variance_matrix(t0-1)->dense());
    for(int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      sparse_scalar_kalman_update(newY[t],
                                  ks.a,
                                  ks.P,
                                  ks.K,
                                  ks.F,
                                  ks.v,
                                  missing,
                                  observation_matrix(t + t0),
                                  observation_variance(t + t0),
                                  *state_transition_matrix(t + t0),
                                  *state_variance_matrix(t + t0));
      ans[t] = ks.v;
    }
    return ans;
  }

}  // namespace BOOM
