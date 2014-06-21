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

#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/StateModels/LocalLevelStateModel.hpp>
#include <distributions.hpp>

namespace BOOM{

  LocalLevelStateModel::LocalLevelStateModel(double sigma)
      : ZeroMeanGaussianModel(sigma),
        state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ConstantMatrix(1, sigma*sigma)),
        initial_state_mean_(1),
        initial_state_variance_(1)
  {}

  LocalLevelStateModel::LocalLevelStateModel(const LocalLevelStateModel &rhs)
      : Model(rhs),
        StateModel(rhs),
        state_transition_matrix_(rhs.state_transition_matrix_),
        state_variance_matrix_(new ConstantMatrix(1, sigsq())),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_)
  {}

  LocalLevelStateModel * LocalLevelStateModel::clone()const{
    return new LocalLevelStateModel(*this);}

  void LocalLevelStateModel::observe_state(const ConstVectorView then,
                                           const ConstVectorView now,
                                           int time_now){
    double current_level = now[0];
    double previous_level = then[0];
    double diff = current_level - previous_level;
    suf()->update_raw(diff);
  }

  uint LocalLevelStateModel::state_dimension()const{return 1;}

  void LocalLevelStateModel::simulate_state_error(VectorView eta, int)const{
    eta[0] = rnorm(0, sigma());
  }

  void LocalLevelStateModel::simulate_initial_state(VectorView eta)const{
    eta[0] = rnorm(initial_state_mean_[0],
                   sqrt(initial_state_variance_(0,0)));
  }

  Ptr<SparseMatrixBlock> LocalLevelStateModel::state_transition_matrix(int)const{
    return state_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> LocalLevelStateModel::state_variance_matrix(int)const{
    return state_variance_matrix_;
  }

  SparseVector LocalLevelStateModel::observation_matrix(int)const{
    SparseVector ans(1);
    ans[0] = 1;
    return ans;
  }

  Vec LocalLevelStateModel::initial_state_mean()const{
    return initial_state_mean_;
  }

  Spd LocalLevelStateModel::initial_state_variance()const{
    return initial_state_variance_;
  }

  void LocalLevelStateModel::set_initial_state_mean(const Vec &m){
    initial_state_mean_ = m;
  }

  void LocalLevelStateModel::set_initial_state_mean(double m){
    initial_state_mean_[0] = m;
  }

  void LocalLevelStateModel::set_initial_state_variance(const Spd &v){
    initial_state_variance_ = v;
  }

  void LocalLevelStateModel::set_initial_state_variance(double v){
    initial_state_variance_(0,0) = v;
  }

  void LocalLevelStateModel::set_sigsq(double sigsq){
    ZeroMeanGaussianModel::set_sigsq(sigsq);
    state_variance_matrix_->set_value(sigsq);
  }

}
