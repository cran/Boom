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
#ifndef BOOM_STATE_SPACE_MODEL_BASE_HPP_
#define BOOM_STATE_SPACE_MODEL_BASE_HPP_
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <Models/StateSpace/Filters/ScalarKalmanStorage.hpp>
#include <Models/StateSpace/PosteriorSamplers/SufstatManager.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Vector.hpp>

#include <boost/scoped_ptr.hpp>

namespace BOOM{

  class StateSpaceModelBase : public CompositeParamPolicy{
   public:
    StateSpaceModelBase();
    StateSpaceModelBase(const StateSpaceModelBase &rhs);
    StateSpaceModelBase * clone() const override = 0;

    // The number of time points in the training data.
    virtual int time_dimension() const = 0;

    // Number of elements in the state vector at a single time point.
    virtual int state_dimension() const;

    // The number of state models.  Presently, a fixed regression
    // model does not count as a state model, nor does a Harvey
    // Cumulator.  This may change in the future.
    int nstate() const {return state_models_.size();}

    // Variance of observed data y[t], given state alpha[t].  Durbin
    // and Koopman's H.
    virtual double observation_variance(int t) const = 0;

    // Returns y[t], after adjusting for regression effects that are
    // not included in the state vector.  This is the value that the
    // time series portion of the model is supposed to describe.  If
    // there are no regression effects, or if the state contains a
    // RegressionStateModel this is literally y[t].  If there are
    // regression effects it is y[t] - beta * x[t].  If y[t] is
    // missing then infinity() is returned.
    virtual double adjusted_observation(int t) const = 0;

    // Returns true if observation t is missing, and false otherwise.
    virtual bool is_missing_observation(int t) const = 0;

    // Returns a pointer to the model responsible for the observation
    // variance.
    virtual Model * observation_model() = 0;
    virtual const Model * observation_model() const = 0;

    // Use Durbin and Koopman's method of forward filtering-backward
    // sampling.  (1) Sample the state vector and an auxiliary y
    // vector from the model.  (2) Subtract the expected value of the
    // state given the simulated y and add the expected value of the
    // state given the observed y.
    virtual void impute_state();

    // The 'observe_state' functions compute the contribution to the
    // complete data sufficient statistics (for the observation and
    // state models) once the state at time 't' has been imputed.
    virtual void observe_state(int t);

    // The initial state can be treated specially, though the default
    // for this function is a no-op.
    virtual void observe_initial_state();

    // This is a hook that tells the observation model to update its
    // sufficient statisitcs now that the state for time t has been
    // observed.
    virtual void observe_data_given_state(int t) = 0;

    // Returns the vector of one step ahead prediction errors for the
    // training data.
    Vector one_step_prediction_errors() const;

    // Clears sufficient statistics for state models and for
    // the client model describing observed data given state.
    virtual void clear_client_data();

    // Add structure to the state portion of the model.  This is for
    // local linear trend and different seasonal effects.  It is not
    // for regression, which this class will handle separately.  The
    // state model should be initialized (including the model for the
    // initial state), and have its learning method (e.g. posterior
    // sampler) set prior to being added using add_state.
    void add_state(Ptr<StateModel>);

    // Durbin and Koopman's T[t] built from state models.
    virtual const SparseKalmanMatrix * state_transition_matrix(int t) const;

    // Durbin and Koopman's Z[t].transpose() built from state models.
    virtual SparseVector observation_matrix(int t) const;

    // Durbin and Koopman's RQR^T.  Built from state models, often
    // less than full rank.
    virtual const SparseKalmanMatrix * state_variance_matrix(int t) const;

    double log_likelihood() const;

    // filter() evaluates log likelihood and computes the final values
    // a[t+1] and P[t+1] needed for future forecasting.
    const ScalarKalmanStorage & filter() const;

    virtual void simulate_initial_state(VectorView v) const;
    virtual Vector simulate_initial_state() const;

    // Simulates the value of the state vector for the current time
    // period, t, given the value of state at the previous time
    // period, t-1.
    // Args:
    //   last:  Value of state at time t-1.
    //   next:  VectorView to be filled with state at time t.
    //   t:  The time index of 'next'.
    void simulate_next_state(const ConstVectorView last,
                             VectorView next,
                             int t) const;
    Vector simulate_next_state(const Vector &current_state, int t) const;

    // Simulates the error for the state at time t+1.  (Using the
    // notation of Durbin and Koopman, this uses the model matrices
    // indexed as t.)
    //
    // Returns a vector of size state_dimension().  If the model
    // matrices are not full rank then some elements of this vector
    // will be deterministic functions of other elements.
    virtual Vector simulate_state_error(int t) const;

    // Parameters of initial state distribution, specified in the
    // state models given to add_state.
    virtual Vector initial_state_mean() const;
    virtual SpdMatrix initial_state_variance() const;

    Ptr<StateModel> state_model(int s){return state_models_[s];}
    const Ptr<StateModel> state_model(int s) const {return state_models_[s];}

    bool kalman_filter_is_current() const {return kalman_filter_is_current_;}

    // Sets an observer in 'params' that invalidates the Kalman filter
    // whenever params changes.
    void observe(Ptr<Params> params);

    ConstVectorView final_state() const;
    ConstVectorView state(int t) const;
    const Matrix &state() const;

    // Returns the contributions of each state model to the overall
    // mean of the series.  The outer vector is indexed by state
    // model.  The inner Vector is a time series.
    std::vector<Vector> state_contributions() const;

    // Returns a time series giving the contribution of state model
    // 'which_model' to the overall mean of the series being modeled.
    Vector state_contribution(int which_model) const;

    // If the model contains a regression component, then return the
    // contribution of the regression model to the overall mean of y
    // at each time point.  If there is no regression component then
    // an empty vector is returned.
    virtual Vector regression_contribution() const;
    virtual bool has_regression() const {return false;}

    // Takes the full state vector as input, and returns the component
    // of the state vector belonging to state model s.
    VectorView state_component(Vector &state, int s) const;
    VectorView state_component(VectorView &state, int s) const;
    ConstVectorView state_component(const ConstVectorView &state, int s) const;

    // Returns a matrix giving contributions of state model s.  Each
    // row is a time series corresponding to one dimension of state
    // for state model s.
    Matrix full_time_series_state_component(int s) const;

    // Sets the behavior of all client state models to 'behavior.'
    void set_state_model_behavior(StateModel::Behavior behavior);

    // The next two member functions are mainly used for debugging a
    // simulation.  You can 'permanently_set_state' to the 'true'
    // state value, then see if the model recovers the parameters.
    // These functions are unlikely to be useful in an actual data
    // analysis.
    void permanently_set_state(const Matrix &m);
    void observe_fixed_state();

    // This function is designed to be called in the constructor of a
    // PosteriorSampler managing this object.
    //
    // Args:
    //   observer: A functor, typically containing a pointer to a
    //     PosteriorSampler managing both this object and a set of
    //     complete data sufficient statistics.  Calling the functor
    //     with a non-negative integer t indicates that observation t
    //     has changed, so the complete data sufficient statistics
    //     should be updated.  Calling the functor with a negative
    //     integer is a signal to reset the complete data sufficient
    //     statistics.
    void register_data_observer(StateSpace::SufstatManagerBase *observer);

   protected:
    // Send a signal to all data observers (typically just 1) that the
    // complete data for observation t has changed.
    //
    // This function is designed to be called as part of the
    // implementation for observe_data_given_state.
    void signal_complete_data_change(int t);

   private:
    void check_kalman_storage(std::vector<LightKalmanStorage> &);
    void initialize_final_kalman_storage() const;
    void kalman_filter_is_not_current(){
      kalman_filter_is_current_ = false;
      mcmc_kalman_storage_is_current_ = false;
    }

    // Send a signal to all data observers (typically just 1) that the
    // complete data sufficient statistics should be reset.
    void signal_complete_data_reset();

    // These are the steps needed to implement impute_state().
    void resize_state();
    void simulate_forward();
    double simulate_adjusted_observation(int t);
    Vector smooth_disturbances(std::vector<LightKalmanStorage> &);
    void propagate_disturbances(const Vector &r0_plus,
                                const Vector &r0_hat,
                                bool observe = true);

    //----------------------------------------------------------------------
    // data starts here
    std::vector<Ptr<StateModel> > state_models_;

    // constructors set state_dimension to zero.  It is incremented
    // during calls to add_state
    int state_dimension_;

    // state_positions_[s] is the index in the state vector where the
    // state for state_models_[s] begins.  There will be one more
    // entry in this vector than the number of state models.  The last
    // entry can be ignored.
    std::vector<int> state_positions_;

    // Workspace for impute_state.  No need to worry about this in the
    // constructor, as it will be initialized as needed.
    Matrix state_;
    Vector a_;
    SpdMatrix P_;
    std::vector<LightKalmanStorage> kalman_storage_;

    // state_is_fixed_ is for use in debugging.  If it is set then the
    // state will be held constant in the data imputation.
    bool state_is_fixed_;
    bool mcmc_kalman_storage_is_current_;

    // Supplemental storage is for filtering simulated observations
    // for Durbin and Koopman's simulation smoother.
    Vector supplemental_a_;
    SpdMatrix supplemental_P_;
    std::vector<LightKalmanStorage> supplemental_kalman_storage_;

    // final_kalman_storage_ holds the output of the Kalman filter.
    // It is for situations where we don't need to store the whole
    // filter, so the name 'final' refers to the fact that it is the
    // last kalman_storage you end up with after running the kalman
    // recursions.  The kalman_filter_is_current_ flag keeps track of
    // whether the parameters have changed since the last time the
    // filter was run.  It must be set by the constructor.  The others
    // will be managed by filter().
    mutable ScalarKalmanStorage final_kalman_storage_;
    mutable double log_likelihood_;
    mutable bool kalman_filter_is_current_;

    mutable boost::scoped_ptr<BlockDiagonalMatrix>
    default_state_transition_matrix_;

    mutable boost::scoped_ptr<BlockDiagonalMatrix>
    default_state_variance_matrix_;

    // Data observers exist so that changes to the (latent) data made
    // by the model can be incorporated by PosteriorSampler classes
    // keeping track of complete data sufficient statistics.  Gaussian
    // models do not need observers, but mixtures of Gaussians
    // typically do.
    std::vector<StateSpace::SufstatManager> data_observers_;
  };

}

#endif // BOOM_STATE_SPACE_MODEL_BASE_HPP_
