#ifndef BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
/*
  Copyright (C) 2008 Steven L. Scott

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
#include <Models/ZeroMeanGaussianModel.hpp>

namespace BOOM{

  class LocalLevelStateModel
      : public StateModel,
        public ZeroMeanGaussianModel
  {
   public:
    LocalLevelStateModel(double sigma=1);
    LocalLevelStateModel(const LocalLevelStateModel &rhs);
    virtual LocalLevelStateModel * clone()const;
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

    void set_initial_state_mean(double m);
    void set_initial_state_mean(const Vec & m);
    void set_initial_state_variance(const Spd &v);
    void set_initial_state_variance(double v);

    virtual void set_sigsq(double sigsq);
   private:
    Ptr<IdentityMatrix> state_transition_matrix_;
    Ptr<ConstantMatrix> state_variance_matrix_;
    Vec initial_state_mean_;
    Spd initial_state_variance_;
  };

}

#endif// BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
