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

#include <Models/StateSpace/StateModels/RegressionStateModel.hpp>

namespace BOOM{

  RegressionStateModel::RegressionStateModel(Ptr<RegressionModel> rm)
      : reg_(rm),
        transition_matrix_(new IdentityMatrix(1)),
        error_variance_(new ZeroMatrix(1))
  {}

  // The copy constructor copies pointers to private data.  Only reg_
  // is controversial, as all the others are the same across all
  // classes.  They could easily be static members.
  RegressionStateModel::RegressionStateModel(const RegressionStateModel &rhs)
      : StateModel(rhs),
        reg_(rhs.reg_),
        transition_matrix_(rhs.transition_matrix_),
        error_variance_(rhs.error_variance_)
  {}

  RegressionStateModel * RegressionStateModel::clone()const{
    return new RegressionStateModel(*this);}

  void RegressionStateModel::clear_data(){
    reg_->suf()->clear(); }

  // This function is a no-op.  The responsibility for observing state
  // lies with the state space model that owns it.
  void RegressionStateModel::observe_state(const ConstVectorView then,
                                           const ConstVectorView now,
                                           int time_now){}

  uint RegressionStateModel::state_dimension()const{return 1;}

  void RegressionStateModel::simulate_state_error(VectorView eta, int t)const{
    eta[0] = 0; }

  void RegressionStateModel::simulate_initial_state(VectorView eta)const{
    eta[0] = 1;}

  Ptr<SparseMatrixBlock> RegressionStateModel::state_transition_matrix(int t)const{
    return transition_matrix_;
  }

  Ptr<SparseMatrixBlock> RegressionStateModel::state_variance_matrix(int)const{
    return error_variance_;
  }

  SparseVector RegressionStateModel::observation_matrix(int t)const{
    double eta = reg_->predict(reg_->dat()[t]->x());
    SparseVector ans(1);
    ans[0] = eta;
    return ans;
  }

  Vec RegressionStateModel::initial_state_mean()const{
    return Vec(1, 1.0);
  }

  Spd RegressionStateModel::initial_state_variance()const{
    return Spd(1, 0.0);
  }
}
