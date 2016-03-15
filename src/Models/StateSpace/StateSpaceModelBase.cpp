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

#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <distributions.hpp>
#include <Models/StateSpace/StateSpaceModelBase.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <cpputil/report_error.hpp>
#include <LinAlg/SubMatrix.hpp>

namespace BOOM{

  typedef StateSpaceModelBase SSMB;

  //----------------------------------------------------------------------
  SSMB::StateSpaceModelBase()
      : state_dimension_(0),
        state_positions_(1, 0),
        state_is_fixed_(false),
        mcmc_kalman_storage_is_current_(false),
        kalman_filter_is_current_(false),
        default_state_transition_matrix_(new BlockDiagonalMatrix),
        default_state_variance_matrix_(new BlockDiagonalMatrix)
  {}

  //----------------------------------------------------------------------
  SSMB::StateSpaceModelBase(const SSMB &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        state_dimension_(0),
        state_positions_(1, 0),
        state_is_fixed_(rhs.state_is_fixed_),
        mcmc_kalman_storage_is_current_(false),
        kalman_filter_is_current_(false),
        default_state_transition_matrix_(new BlockDiagonalMatrix),
        default_state_variance_matrix_(new BlockDiagonalMatrix)
  {
    for (int s = 0; s < rhs.nstate(); ++s) {
      add_state(rhs.state_model(s)->clone());
    }
    if (state_is_fixed_) state_ = rhs.state_;
  }

  //----------------------------------------------------------------------
  void SSMB::impute_state() {
    set_state_model_behavior(StateModel::MIXTURE);
    if (state_is_fixed_) {
      observe_fixed_state();
    } else {
      resize_state();
      clear_client_data();
      simulate_forward();
      Vector r0_sim = smooth_disturbances(kalman_storage_);
      Vector r0_obs = smooth_disturbances(supplemental_kalman_storage_);
      propagate_disturbances(r0_sim, r0_obs, true);
    }
  }

  //----------------------------------------------------------------------
  // Ensure that state_ is large enough to hold the results of
  // impute_state().
  void SSMB::resize_state() {
    if (nrow(state_) != state_dimension()
       || ncol(state_) != time_dimension()) {
      state_.resize(state_dimension(), time_dimension());
    }
    for (int s = 0; s < state_models_.size(); ++s) {
      state_models_[s]->observe_time_dimension(time_dimension());
    }
  }

  //----------------------------------------------------------------------
  // Simulate alpha_+ and y_* = y - y_+.  While simulating y_*,
  // feed it into the light (no storage for P) Kalman filter.  The
  // simulated state is stored in state_, while kalman_storage_ holds
  // the output of the Kalman filter.
  //
  // y_+ and alpha_+ will be simulated in parallel with
  // Kalman filtering and disturbance smoothing of y, and the results
  // will be subtracted to compute y_*.
  void SSMB::simulate_forward() {
    check_kalman_storage(kalman_storage_);
    check_kalman_storage(supplemental_kalman_storage_);
    log_likelihood_ = 0;
    for (int t = 0; t < time_dimension(); ++t) {
      // simulate_state at time t
      if (t == 0) {
        simulate_initial_state(state_.col(0));
        a_ = initial_state_mean();
        P_ = initial_state_variance();
        supplemental_a_ = a_;
        supplemental_P_ = P_;
      }else{
        simulate_next_state(state_.col(t-1), state_.col(t), t);
      }
      double y_sim = simulate_adjusted_observation(t);
      sparse_scalar_kalman_update(
          y_sim,
          a_,
          P_,
          kalman_storage_[t].K,
          kalman_storage_[t].F,
          kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          *state_transition_matrix(t),
          *state_variance_matrix(t));
        ////////////////////////
        // TODO(stevescott): The actual one step ahead prediction
        // errors are being stored in supplemental_kalman_storage_,
        // and not kalman_storage_.  We should eventually keep the
        // prediction errors in the right place.
      log_likelihood_ += sparse_scalar_kalman_update(
          adjusted_observation(t),
          supplemental_a_,
          supplemental_P_,
          supplemental_kalman_storage_[t].K,
          supplemental_kalman_storage_[t].F,
          supplemental_kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          (*state_transition_matrix(t)),
          (*state_variance_matrix(t)));

      // The Kalman update sets a_ to a[t+1] and P to P[t+1], so they
      // will be current for the next iteration.
    }
    mcmc_kalman_storage_is_current_ = true;
  }

  //----------------------------------------------------------------------
  double SSMB::simulate_adjusted_observation(int t) {
    double mu = observation_matrix(t).dot(state_.col(t));
    return rnorm(mu, sqrt(observation_variance(t)));
  }

  //----------------------------------------------------------------------
  // Disturbance smoother replaces Durbin and Koopman's K[t] with
  // r[t].  The disturbance smoother is equation (5) in Durbin and
  // Koopman (2002).
  // TODO(stevescott): make sure you've got t, t-1, and t+1 worked out
  // correctly.
  Vector SSMB::smooth_disturbances(
      std::vector<LightKalmanStorage> &kalman_storage) {
    int n = time_dimension();
    Vector r(state_dimension(), 0.0);
    for (int t = n-1; t>=0; --t) {
      // Upon entry r is r[t].
      // On exit, r is r[t-1] and kalman_storage[t].K is r[t]

      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z[t] * v[t]/F[t] + (T[t]^T - Z[t] * K[t]^T)r[t]
      //        = T[t]^T * r + Z[t] * (v[t]/F[t] - K.dot(r))

      // Some syntactic sugar makes the formulas easier to match up
      // with Durbin and Koopman.
      double v = kalman_storage[t].v;
      double F = kalman_storage[t].F;
      Vector &K(kalman_storage[t].K);
      double coefficient = (v/F) - K.dot(r);

      // Now produce r[t-1]
      Vector rt_1 = state_transition_matrix(t)->Tmult(r);
      observation_matrix(t).add_this_to(rt_1, coefficient);
      K = r;
      r = rt_1;
    }
    return r;
  }

  //----------------------------------------------------------------------
  // After a call to smooth_disturbances() puts r[t] in
  // kalman_storage_[t].K, this function propagates the r's forward to
  // get E(alpha | y), and add it to the simulated state.
  void SSMB::propagate_disturbances(
      const Vector &r0_sim, const Vector & r0_obs, bool observe) {
    // TODO(stevescott): Two linear operations are being performed in
    // parallel.  Can they be replaced by a single linear operation on
    // the difference?
    if (state_.ncol() <= 0) return;
    SpdMatrix P0 = initial_state_variance();
    Vector state_mean_sim = initial_state_mean() + P0*r0_sim;
    Vector state_mean_obs = initial_state_mean() + P0*r0_obs;

    state_.col(0) += state_mean_obs - state_mean_sim;
    if (observe) {
      observe_state(0);
      observe_data_given_state(0);
    }
    for (int t = 1; t < time_dimension(); ++t) {
      state_mean_sim = (*state_transition_matrix(t-1)) * state_mean_sim
          + (*state_variance_matrix(t-1)) * kalman_storage_[t-1].K;
      state_mean_obs = (*state_transition_matrix(t-1)) * state_mean_obs
          + (*state_variance_matrix(t-1)) * supplemental_kalman_storage_[t-1].K;

      state_.col(t).axpy(state_mean_obs - state_mean_sim);
      if (observe) {
        observe_state(t);
        observe_data_given_state(t);
      }
    }
  }

  void SSMB::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
      return;
    }
    const ConstVectorView now(state_.col(t));
    const ConstVectorView then(state_.col(t-1));
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->observe_state(
          state_component(then, s),
          state_component(now, s),
          t);
    }
  }

  void SSMB::observe_initial_state() {
    for (int s = 0; s < nstate(); ++s) {
      ConstVectorView state(state_component(state_.col(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  //----------------------------------------------------------------------
  Vector SSMB::one_step_prediction_errors() const {
    int n = time_dimension();
    Vector errors(n);
    if (n == 0) return errors;

    if (mcmc_kalman_storage_is_current_) {
      for (int i = 0; i < n; ++i) {
        // TODO(stevescott): Clean up this hack by making sure the one
        // step prediction errors are stored in kalman_storage_
        // instead of supplemental_kalman_storage_.
        errors[i] = supplemental_kalman_storage_[i].v;
      }
      return errors;
    }
    log_likelihood_ = 0;
    initialize_final_kalman_storage();
    ScalarKalmanStorage &ks(final_kalman_storage_);

    for (int i = 0; i < n; ++i) {
      double resid = adjusted_observation(i);
      bool missing = is_missing_observation(i);
      log_likelihood_ += sparse_scalar_kalman_update(
          resid,
          ks.a,
          ks.P,
          ks.K,
          ks.F,
          ks.v,
          missing,
          observation_matrix(i),
          observation_variance(i),
          (*state_transition_matrix(i)),
          (*state_variance_matrix(i)));
      errors[i] = ks.v;
    }
    kalman_filter_is_current_ = true;
    return errors;
  }

  //----------------------------------------------------------------------
  void SSMB::clear_client_data() {
    observation_model()->clear_data();
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->clear_data();
    }
    signal_complete_data_reset();
  }

  //----------------------------------------------------------------------
  void SSMB::add_state(Ptr<StateModel> m) {
    ParamPolicy::add_model(m);
    state_models_.push_back(m);
    state_dimension_ += m->state_dimension();
    int next_position = state_positions_.back()
          + m->state_dimension();
    state_positions_.push_back(next_position);
    std::vector<Ptr<Params> > params(m->t());
    for (int i = 0; i < params.size(); ++i) observe(params[i]);
  }

  //----------------------------------------------------------------------
  SparseVector SSMB::observation_matrix(int t) const {
    SparseVector ans;
    for (int s = 0; s < nstate(); ++s) {
      ans.concatenate(state_models_[s]->observation_matrix(t));
    }
    return ans;
  }

  //----------------------------------------------------------------------
  // TODO(stevescott): This and other code involving model matrices is
  // an optimization opportunity.  Test it out to see if
  // precomputation makes sense.
  const SparseKalmanMatrix * SSMB::state_transition_matrix(int t) const {
    // Size comparisons should be made with respect to
    // state_dimension_, not state_dimension() which is virtual.
    if (default_state_transition_matrix_->nrow() != state_dimension_
       || default_state_transition_matrix_->ncol() != state_dimension_) {
      default_state_transition_matrix_->clear();
      for (int s = 0; s < state_models_.size(); ++s) {
        default_state_transition_matrix_->add_block(
            state_models_[s]->state_transition_matrix(t));
      }
    }else{
      // If we're in this block, then the matrix must have been
      // created already, and we just need to update the blocks.
      for (int s = 0; s < state_models_.size(); ++s) {
        default_state_transition_matrix_->replace_block(
            s, state_models_[s]->state_transition_matrix(t));
      }
    }
    return default_state_transition_matrix_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix * SSMB::state_variance_matrix(int t) const {
    default_state_variance_matrix_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      default_state_variance_matrix_->add_block(
          state_models_[s]->state_variance_matrix(t));
    }
    return default_state_variance_matrix_.get();
  }

  //----------------------------------------------------------------------
  int SSMB::state_dimension() const {return state_dimension_;}

  //----------------------------------------------------------------------
  double SSMB::log_likelihood() const {
    filter();
    return log_likelihood_;
  }

  //----------------------------------------------------------------------
  const ScalarKalmanStorage & SSMB::filter() const {
    if (kalman_filter_is_current_) return final_kalman_storage_;
    log_likelihood_ = 0;
    initialize_final_kalman_storage();
    int n = time_dimension();
    if (n == 0) return final_kalman_storage_;
    ScalarKalmanStorage &ks(final_kalman_storage_);

    for (int i = 0; i < n; ++i) {
      double resid = adjusted_observation(i);
      bool missing = is_missing_observation(i);
      log_likelihood_ += sparse_scalar_kalman_update(
          resid,
          ks.a,
          ks.P,
          ks.K,
          ks.F,
          ks.v,
          missing,
          observation_matrix(i),
          observation_variance(i),
          (*state_transition_matrix(i)),
          (*state_variance_matrix(i)));
    }
    kalman_filter_is_current_ = true;
    return final_kalman_storage_;
  }

  //----------------------------------------------------------------------
  Vector SSMB::simulate_initial_state() const {
    Vector ans(state_dimension_);
    simulate_initial_state(VectorView(ans));
    return ans;
  }

  //----------------------------------------------------------------------
  // TODO(stevescott):  test
  void SSMB::simulate_initial_state(VectorView state0) const {
    for (int s = 0; s < state_models_.size(); ++s) {
      state_model(s)->simulate_initial_state(state_component(state0, s));
    }
  }

  //----------------------------------------------------------------------
  // Simulates state for time period t
  void SSMB::simulate_next_state(ConstVectorView last,
                                 VectorView next,
                                 int t) const {
    next= (*state_transition_matrix(t-1)) * last;
    next += simulate_state_error(t-1);
  }

  //----------------------------------------------------------------------
  Vector SSMB::simulate_next_state(const Vector &state,
                                   int t) const {
    Vector ans(state);
    simulate_next_state(ConstVectorView(state),
                        VectorView(ans),
                        t);
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSMB::simulate_state_error(int t) const {
    // simulate N(0, RQR) for the state at time t+1, using the
    // variance matrix at time t.
    Vector ans(state_dimension_, 0);
    for (int s = 0; s < state_models_.size(); ++s) {
      VectorView eta(state_component(ans, s));
      state_model(s)->simulate_state_error(eta, t);
    }
    return ans;
  }
  //----------------------------------------------------------------------
  Vector SSMB::initial_state_mean() const {
    Vector ans;
    for (int s = 0; s < state_models_.size(); ++s) {
      ans.concat(state_models_[s]->initial_state_mean());
    }
    return ans;
  }

  //----------------------------------------------------------------------
  SpdMatrix SSMB::initial_state_variance() const {
    SpdMatrix ans(state_dimension_);
    int lo = 0;
    for (int s = 0; s < state_models_.size(); ++s) {
      Ptr<StateModel> state = state_models_[s];
      int hi = lo + state->state_dimension() - 1;
      SubMatrix block(ans, lo, hi, lo, hi);
      block = state_models_[s]->initial_state_variance();
      lo = hi + 1;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  void SSMB::observe(Ptr<Params> p) {
    boost::function<void(void)>f =
        boost::bind(&SSMB::kalman_filter_is_not_current, this);
    p->add_observer(f);
  }

  //----------------------------------------------------------------------
  ConstVectorView SSMB::final_state() const {
    return state_.last_col();
  }

  //----------------------------------------------------------------------
  ConstVectorView SSMB::state(int t) const {
    return state_.col(t);
  }

  //----------------------------------------------------------------------
  const Matrix &SSMB::state() const {return state_;}

  //----------------------------------------------------------------------
  std::vector<Vector> SSMB::state_contributions() const {
    std::vector<Vector> ans(nstate());
    for (int t = 0; t < time_dimension(); ++t) {
      for (int m = 0; m < nstate(); ++m) {
        ConstVectorView state(state_component(state_.col(t), m));
        ans[m].push_back(state_models_[m]->observation_matrix(t).dot(state));
      }
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSMB::state_contribution(int s) const {
    if (ncol(state_) != time_dimension() ||
       nrow(state_) != state_dimension()) {
      ostringstream err;
      err << "state is the wrong size in "
          << "StateSpaceModelBase::state_contribution" << endl
          << "State contribution matrix has " << ncol(state_) << " columns.  "
          << "Time dimension is " << time_dimension() << "." << endl
          << "State contribution matrix has " << nrow(state_) << " rows.  "
          << "State dimension is " << state_dimension() << "." << endl;
      report_error(err.str());
    }
    Vector ans(time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView state(state_component(state_.col(t), s));
      ans[t] = state_model(s)->observation_matrix(t).dot(state);
    }
    return ans;
  }

  Vector SSMB::regression_contribution() const {
    return Vector();
  }

  //----------------------------------------------------------------------
  VectorView SSMB::state_component(Vector &v, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return VectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  VectorView SSMB::state_component(VectorView &v, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return VectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  ConstVectorView SSMB::state_component(const ConstVectorView &v, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return ConstVectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  Matrix SSMB::full_time_series_state_component(int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    ConstSubMatrix contribution(
        state_, start, start + size - 1, 0, time_dimension() - 1);
    return contribution.to_matrix();
  }

  //----------------------------------------------------------------------
  void SSMB::permanently_set_state(const Matrix &state) {
    if ((ncol(state) != time_dimension()) ||
       (nrow(state) != state_dimension())) {
      ostringstream err;
      err << "Wrong dimension of 'state' in "
          << "StateSpaceModelBase::permanently_set_state()."
          << "Argument was " << nrow(state) << " by " << ncol(state)
          << ".  Expected " << state_dimension() << " by "
          << time_dimension() << "." << endl;
      report_error(err.str());
    }
    state_is_fixed_ = true;
    state_ = state;
  }

  //----------------------------------------------------------------------
  void SSMB::observe_fixed_state() {
    clear_client_data();
    for (int t = 0; t < time_dimension(); ++t) {
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  //----------------------------------------------------------------------
  void SSMB::set_state_model_behavior(StateModel::Behavior behavior) {
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->set_behavior(behavior);
    }
  }

  //----------------------------------------------------------------------
  void SSMB::check_kalman_storage(
      std::vector<LightKalmanStorage> &kalman_storage) {
    bool ok = true;
    if (kalman_storage.size() < time_dimension()) {
      kalman_storage.reserve(time_dimension());
      ok = false;
    }

    if (!kalman_storage.empty()) {
      if (kalman_storage[0].K.size() != state_dimension()) {
        kalman_storage.clear();
        ok = false;
      }
    }

    if (!ok) {
      for (int t = kalman_storage.size(); t < time_dimension(); ++t) {
        LightKalmanStorage s(state_dimension());
        kalman_storage.push_back(s);
      }
    }
  }

  //----------------------------------------------------------------------
  void SSMB::initialize_final_kalman_storage() const {
    ScalarKalmanStorage &ks(final_kalman_storage_);
    ks.a = initial_state_mean();
    ks.P = initial_state_variance();
    ks.F = observation_matrix(0).sandwich(ks.P) + observation_variance(0);
  }

  void SSMB::register_data_observer(StateSpace::SufstatManagerBase *smb) {
    data_observers_.push_back(StateSpace::SufstatManager(smb));
  }

  // Send a signal to any object observing this model's data that
  // observation t has changed.
  void SSMB::signal_complete_data_change(int t) {
    for (int i = 0; i< data_observers_.size(); ++i) {
      data_observers_[i].update_complete_data_sufficient_statistics(t);
    }
  }

  // Send a signal to any observers of this model's data that the
  // complete data sufficient statistics should be reset.
  void SSMB::signal_complete_data_reset() {
    for (int i = 0; i < data_observers_.size(); ++i) {
      data_observers_[i].clear_complete_data_sufficient_statistics();
    }
  }


}  // namespace BOOM
