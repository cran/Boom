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

#ifndef BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_
#define BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_

#include <Models/ZeroMeanMvnModel.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>

namespace BOOM{

  //  mu[t+1] = mu[t] + delta[t] + u[t]
  //  delta[t+1] = delta[t] + v[t]
  class LocalLinearTrendStateModel
      : public ZeroMeanMvnModel,
        public StateModel
  {
   public:
    LocalLinearTrendStateModel();
    LocalLinearTrendStateModel(const LocalLinearTrendStateModel &rhs);
    virtual LocalLinearTrendStateModel * clone()const;

    virtual void observe_state(const ConstVectorView then,
                               const ConstVectorView now,
                               int time_now);
    virtual uint state_dimension()const{return 2;}

    virtual void simulate_state_error(VectorView eta, int t)const;

    virtual Ptr<SparseMatrixBlock> state_transition_matrix(int t)const;
    virtual Ptr<SparseMatrixBlock> state_variance_matrix(int t)const;

    virtual SparseVector observation_matrix(int t)const;

    virtual Vec initial_state_mean()const;
    void set_initial_state_mean(const Vec &v);
    virtual Spd initial_state_variance()const;
    void set_initial_state_variance(const Spd &V);

    virtual void set_Sigma(const Spd &sigma);
   private:
    void check_dim(const ConstVectorView &)const;

    SparseVector observation_matrix_;
    Ptr<LocalLinearTrendMatrix> state_transition_matrix_;
    Ptr<DenseSpd> state_variance_matrix_;
    Vec initial_state_mean_;
    Spd initial_state_variance_;
  };


}
#endif // BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_
