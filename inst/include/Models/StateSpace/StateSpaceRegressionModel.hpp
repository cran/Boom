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
#include <LinAlg/Types.hpp>

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
    StateSpaceRegressionModel(const Vec &y,
                              const Mat &X,
                              const std::vector<bool> &observed=
                              std::vector<bool>());

    StateSpaceRegressionModel(const StateSpaceRegressionModel &rhs);
    StateSpaceRegressionModel * clone()const;

    virtual int time_dimension()const{return dat().size();}

    // Variance of observed data y[t], given state alpha[t].  Durbin
    // and Koopman's H.
    virtual double observation_variance(int t)const;

    virtual double adjusted_observation(int t)const;
    virtual bool is_missing_observation(int t)const;
    virtual RegressionModel * observation_model(){
      return regression_.get();}
    virtual const RegressionModel * observation_model()const{
      return regression_.get(); }

    virtual void observe_data_given_state(int t);

    // Forecast the next nrow(newX) time steps given the current data,
    // using the Kalman filter.  The first column of Mat is the mean
    // of the forecast.  The second column is the standard errors.
    Mat forecast(const Mat &newX)const;

    // Simulate the next nrow(newX) time periods, given current
    // parameters and state.
    Vec simulate_forecast(const Mat &newX, const Vec &final_state);
    Vec simulate_forecast(const Mat &newX);

    // Contribution of the regression model to the overall mean of y
    // at each time point.
    Vec regression_contribution()const;

    // Returns the vector of one-step-ahead prediction errors from a
    // holdout sample.
    Vec one_step_holdout_prediction_errors(const Mat &newX,
                                           const Vec &newY,
                                           const Vec &final_state)const;

    Ptr<RegressionModel> regression_model(){return regression_;}
    const Ptr<RegressionModel> regression_model()const{return regression_;}

    // Need to overload add_data so that x's can be shared with the
    // regression model.
    virtual void add_data(Ptr<Data> dp);
    virtual void add_data(Ptr<RegressionData> dp);
   private:
    // The regression model holds the regression coefficients and the
    // observation error variance.
    Ptr<RegressionModel> regression_;

    // Initialization work common to several constructors
    void setup();
  };


}

#endif // BOOM_STATE_SPACE_REGRESSION_HPP_
