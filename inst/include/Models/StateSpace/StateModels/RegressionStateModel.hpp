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

#ifndef BOOM_REGRESSION_STATE_MODEL_HPP_
#define BOOM_REGRESSION_STATE_MODEL_HPP_

#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/Glm/RegressionModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Policies/NullDataPolicy.hpp>

namespace BOOM{

  // A StateModel for a homogeneous regression component.
  // 'Homogeneous' means that the regression coefficients remain
  // constant over time.  They can be parameters to be learned from
  // data, but the learning must take place outside of this class,
  // using an external pointer to the privately held regression model.
  // The 'state' is a constant '1' with zero error, and a [1x1]
  // identity matrix for the state transition matrix.  The single
  // entry in the observation matrix is x[t] * beta, the predicted
  // value from the regression at time t.
  class RegressionStateModel
      : public StateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    RegressionStateModel(Ptr<RegressionModel>);
    RegressionStateModel(const RegressionStateModel &rhs);
    virtual RegressionStateModel * clone()const;

    // clears sufficient statistics, but does not erase pointers to data.
    virtual void clear_data();

    // 'observe_state' is a no-op for this class because the state
    // model needs too much information in order to make the necessary
    // observations.  A class that contains a RegressionStateModel
    // should update an externally held pointer to reg_ each time a
    // state vector is observed.
    virtual void observe_state(const ConstVectorView then,
                               const ConstVectorView now,
                               int time_now);

    virtual uint state_dimension()const;

    virtual void simulate_state_error(VectorView eta, int t)const;
    virtual void simulate_initial_state(VectorView eta)const;

    virtual Ptr<SparseMatrixBlock> state_transition_matrix(int t)const;
    virtual Ptr<SparseMatrixBlock> state_variance_matrix(int t)const;
    virtual SparseVector observation_matrix(int t)const;

    virtual Vec initial_state_mean()const;
    virtual Spd initial_state_variance()const;

   private:
    Ptr<RegressionModel> reg_;
    Ptr<IdentityMatrix> transition_matrix_;
    Ptr<ZeroMatrix> error_variance_;

   protected:
    RegressionModel * regression() {return reg_.get();}
    const RegressionModel * regression()const{return reg_.get();}
  };

}
#endif // BOOM_REGRESSION_STATE_MODEL_HPP_
