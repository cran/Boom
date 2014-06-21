#ifndef BOOM_STATE_SPACE_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_STATE_MODEL_HPP
/*
  Copyright (C) 2008-2011 Steven L. Scott

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

#include <Models/ModelTypes.hpp>
#include <LinAlg/VectorView.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <uint.hpp>

namespace BOOM{

  // A StateModel describes the propogation rules for one component of
  // state in a StateSpaceModel.  A StateModel has a transition matrix
  // T, which can be time dependent, an error variance Q, which may be
  // of smaller dimension than T, and a matrix R that can multiply
  // draws from N(0, Q) so that the dimension of RQR^T matches the
  // state dimension.
  class StateModel
      : virtual public Model
  {
   public:
    // Traditional state models are Gaussian, but Bayesian modeling
    // lets you work with conditionally Gaussian models just as
    // easily.  For conditionally Gaussian state models this enum can
    // be used as an argument to determine whether they should be
    // viewed as normal mixtures, or as plain old non-normal marginal
    // models.
    enum Behavior{
      MARGINAL, // e.g. treat the t-distribution like the t-distribution.
      MIXTURE   // e.g. treat the t-distribution like a normal mixture.
    };

    virtual ~StateModel(){}
    virtual StateModel * clone()const=0;

    // Some state models need to know the maximum value of t so they
    // can set up space for latent variables, etc.  Many state models
    // do not need this capability, so the default implementation is a
    // no-op.
    virtual void observe_time_dimension(int max_time) {}

    // Add the relevant information from the state vector to the
    // complete data sufficient statistics for this model.  This is
    // often a difference between the current and next state vectors.
    virtual void observe_state(const ConstVectorView then,
                               const ConstVectorView now,
                               int time_now)=0;

    // Many models won't be able to do anything with an initial state,
    // so the default implementation is a no-op.
    virtual void observe_initial_state(const ConstVectorView &state);

    // The dimension of the state vector.
    virtual uint state_dimension()const=0;

    virtual void simulate_state_error(VectorView eta, int t)const = 0;
    virtual void simulate_initial_state(VectorView eta)const;

    virtual Ptr<SparseMatrixBlock> state_transition_matrix(int t)const = 0;
    virtual Ptr<SparseMatrixBlock> state_variance_matrix(int t)const = 0;

    //  For now, limit models to have constant observation matrices.
    //  This will prevent true DLM's with coefficients in the Kalman
    //  filter, because this is where the x's would go, but we'll need
    //  a different API for that case anyway.
    virtual SparseVector observation_matrix(int t)const = 0;

    virtual Vec initial_state_mean()const = 0;
    virtual Spd initial_state_variance()const = 0;

    // Some state models can behave differently in different contexts.
    // E.g. they can be viewed as conditionally normal when fitting,
    // but as T or normal mixtures when forecasting.  These virtual
    // functions control how the state models swtich between roles.
    // The default behavior at construction should be
    // 'set_conditional_behavior', where a state model will behave as
    // conditionally Gaussian given an appropriate set of latent
    // variables.
    //
    // Because the traditional state models are actually Gaussian
    // (instead of simply conditionally Gaussian), the default
    // behavior for these member functions is a no-op.
    virtual void set_behavior(Behavior) {}
  };
}

#endif// BOOM_STATE_SPACE_STATE_MODEL_HPP
