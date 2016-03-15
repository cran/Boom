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
#ifndef BOOM_STATE_SPACE_REGRESSION_HPP_
#define BOOM_STATE_SPACE_REGRESSION_HPP_

#include <Models/StateSpace/StateSpaceModelBase.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <Models/StateSpace/Filters/ScalarKalmanStorage.hpp>

#include <Models/Glm/Glm.hpp>
#include <Models/Glm/RegressionModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

#include <LinAlg/Matrix.hpp>
#include <LinAlg/Vector.hpp>

#include <vector>

namespace BOOM{

  // A contemporaneous regression model, where y[t] =
  // beta.dot(X.row(t)) + state space.
  class StateSpaceRegressionModel
      : public StateSpaceModelBase,
        public IID_DataPolicy<RegressionData>,
        public PriorPolicy
  {
   public:
    // xdim is the dimension of the x's in the regression part of the
    // model.
    StateSpaceRegressionModel(int xdim);

    // y is the time series of observations.  X is the design matrix,
    // with rows contemporaneous to y.  If some of the y's are
    // missing, use 'observed' to indicate which are observed.  The X's
    // must be fully observed.
    StateSpaceRegressionModel(const Vector &y,
                              const Matrix &X,
                              const std::vector<bool> &observed=
                              std::vector<bool>());

    StateSpaceRegressionModel(const StateSpaceRegressionModel &rhs);
    StateSpaceRegressionModel * clone()const override;

    int time_dimension() const override {return dat().size();}

    // Variance of observed data y[t], given state alpha[t].  Durbin
    // and Koopman's H.
    double observation_variance(int t) const override;

    double adjusted_observation(int t) const override;
    bool is_missing_observation(int t) const override;
    RegressionModel * observation_model() override {
      return regression_.get(); }
    const RegressionModel * observation_model() const override {
      return regression_.get(); }

    void observe_data_given_state(int t) override;

    // Forecast the next nrow(newX) time steps given the current data,
    // using the Kalman filter.  The first column of Matrix is the mean
    // of the forecast.  The second column is the standard errors.
    Matrix forecast(const Matrix &newX)const;

    // Simulate the next nrow(newX) time periods, given current
    // parameters and state.
    Vector simulate_forecast(const Matrix &newX, const Vector &final_state);
    Vector simulate_forecast(const Matrix &newX);

    // Contribution of the regression model to the overall mean of y
    // at each time point.
    Vector regression_contribution()const override;
    bool has_regression() const override {return true;}

    // Returns the vector of one-step-ahead prediction errors from a
    // holdout sample.
    Vector one_step_holdout_prediction_errors(const Matrix &newX,
                                              const Vector &newY,
                                              const Vector &final_state)const;

    Ptr<RegressionModel> regression_model(){return regression_;}
    const Ptr<RegressionModel> regression_model()const{return regression_;}

    // Need to override add_data so that x's can be shared with the
    // regression model.
    void add_data(Ptr<Data> dp) override;
    void add_data(Ptr<RegressionData> dp) override;

   private:
    // The regression model holds the regression coefficients and the
    // observation error variance.
    Ptr<RegressionModel> regression_;

    // Initialization work common to several constructors
    void setup();
  };

}  // namespace BOOM

#endif // BOOM_STATE_SPACE_REGRESSION_HPP_
