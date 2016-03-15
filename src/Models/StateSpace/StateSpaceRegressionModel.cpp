/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <Models/StateSpace/StateSpaceRegressionModel.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <Models/DataTypes.hpp>
#include <distributions.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

namespace BOOM{

  typedef StateSpaceRegressionModel SSRM;

  void SSRM::setup() {
    observe(regression_->coef_prm());
    observe(regression_->Sigsq_prm());
    regression_->only_keep_sufstats(true);
    ParamPolicy::add_model(regression_);
  }

  SSRM::StateSpaceRegressionModel(int xdim)
      : regression_(new RegressionModel(xdim))
  {
    setup();
    // Note that in this constructor the regression model will still
    // need to have data added, so we can't call fix_xtx().  This
    // means that the xtx matrix will be re-computed with each trip
    // through the data.
  }

  SSRM::StateSpaceRegressionModel(const Vector &y, const Matrix &X,
                                  const std::vector<bool> &observed)
      : regression_(new RegressionModel(ncol(X)))
  {
    setup();
    int n = y.size();
    if (nrow(X) != n) {
      ostringstream msg;
      msg << "X and y are incompatible in constructor for "
          << "StateSpaceRegressionModel." << endl
          << "length(y) = " << n << endl
          << "nrow(X) = " << nrow(X) << endl;
      report_error(msg.str());
    }

    for (int i = 0; i < n; ++i) {
      NEW(RegressionData, dp)(y[i], X.row(i));
      if (!(observed.empty()) && !observed[i]) {
        dp->set_missing_status(Data::partly_missing);
      }
      add_data(dp);
    }

    // The cast is necessary because the regression model stores a Ptr
    // to a base class that does not supply fix_xtx();
    regression_->suf().dcast<NeRegSuf>()->fix_xtx();
  }

  SSRM::StateSpaceRegressionModel(const SSRM &rhs)
      : Model(rhs),
        StateSpaceModelBase(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        regression_(rhs.regression_->clone())
  {
    setup();
  }

  SSRM * SSRM::clone() const {return new SSRM(*this);}

  void SSRM::add_data(Ptr<Data> dp) { add_data(DAT(dp)); }

  void SSRM::add_data(Ptr<RegressionData> dp) {
    DataPolicy::add_data(dp);
    regression_->add_data(dp);
  }

  double SSRM::observation_variance(int) const {
    return regression_->sigsq();
  }

  double SSRM::adjusted_observation(int t) const {
    Ptr<RegressionData> dp = dat()[t];
    return dp->y() - regression_->predict(dp->x());
  }

  bool SSRM::is_missing_observation(int t) const {
    return dat()[t]->missing() != Data::observed;
  }

  void SSRM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      Ptr<RegressionData> dp(dat()[t]);
      double state_mean = observation_matrix(t).dot(state(t));
      regression_->suf()->add_mixture_data(
          dp->y() - state_mean, dp->x(), 1.0);
    }
  }

  Matrix SSRM::forecast(const Matrix &newX) const {
    ScalarKalmanStorage ks = filter();
    Matrix ans(nrow(newX), 2);
    int t0 = dat().size();
    for (int t = 0; t < nrow(ans); ++t) {
      ans(t,0) = regression_->predict(newX.row(t))
          + observation_matrix(t + t0).dot(ks.a);
      sparse_scalar_kalman_update(0,  // y is missing, so fill in a dummy value
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

  // TODO(stevescott):  test simulate_forecast
  Vector SSRM::simulate_forecast(const Matrix &newX,
                                 const Vector &final_state) {
    StateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(nrow(newX));
    const std::vector<Ptr<RegressionData> > &data(dat());
    int t0 = data.size();
    Vector state = final_state;
    for (int t = 0; t < ans.size(); ++t) {
      state = simulate_next_state(state, t + t0);
      ans[t] = rnorm(observation_matrix(t + t0).dot(state),
                     sqrt(observation_variance(t + t0)));
      ans[t] += regression_->predict(newX.row(t));
    }
    return ans;
  }

  Vector SSRM::simulate_forecast(const Matrix &newX) {
    StateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    ScalarKalmanStorage kalman_storage = filter();
    // The Kalman filter produces the forecast distribution for the
    // next time period.  Since the observed data goes from 0 to t-1,
    // kalman_storage contains the forecast distribution for time t.
    Vector final_state = rmvn_robust(kalman_storage.a, kalman_storage.P);
    return simulate_forecast(newX, final_state);
  }

  Vector SSRM::one_step_holdout_prediction_errors(
      const Matrix &newX,
      const Vector &newY,
      const Vector &final_state) const {
    if (nrow(newX) != length(newY)) {
      report_error("X and Y do not match in StateSpaceRegressionModel::"
                   "one_step_holdout_prediction_errors");
    }

    Vector ans(nrow(newX));
    const std::vector<Ptr<RegressionData> > &data(dat());
    int t0 = data.size();
    ScalarKalmanStorage ks(state_dimension());
    ks.a = *state_transition_matrix(t0-1) * final_state;
    ks.P = SpdMatrix(state_variance_matrix(t0-1)->dense());

    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      sparse_scalar_kalman_update(
          newY[t] - regression_model()->predict(newX.row(t)),
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

  Vector SSRM::regression_contribution() const {
    Vector ans(time_dimension());
    const std::vector<Ptr<RegressionData> > &data(dat());
    for (int i = 0; i < data.size(); ++i) {
      ans[i] = regression_model()->predict(data[i]->x());
    }
    return ans;
  }

}  // namespace BOOM
