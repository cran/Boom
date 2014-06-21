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

#ifndef BOOM_LOCAL_LINEAR_TREND_MEAN_REVERTING_SLOPE_STATE_MODEL_HPP_
#define BOOM_LOCAL_LINEAR_TREND_MEAN_REVERTING_SLOPE_STATE_MODEL_HPP_
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>
#include <Models/TimeSeries/NonzeroMeanAr1Model.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <Models/TimeSeries/NonzeroMeanAr1Model.hpp>
namespace BOOM{

  // The state transition matrix for the
  // LocalLinearTrendMeanRevertingSlopeMatrix is
  //  1   1   0
  //  0  phi (1-phi)
  //  0   0   1
  class LocalLinearTrendMeanRevertingSlopeMatrix
      : public SparseMatrixBlock {
   public:
    LocalLinearTrendMeanRevertingSlopeMatrix(Ptr<UnivParams> phi);

    // Can safely copy with pointer semantics, becasue nothing in this
    // class can change the value of the pointer.
    LocalLinearTrendMeanRevertingSlopeMatrix(
        const LocalLinearTrendMeanRevertingSlopeMatrix &rhs);
    virtual LocalLinearTrendMeanRevertingSlopeMatrix * clone()const;
    virtual int nrow()const{return 3;}
    virtual int ncol()const{return 3;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const;
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const;
    virtual void multiply_inplace(VectorView x)const;
    virtual void add_to(SubMatrix block)const;
    virtual Mat dense()const;
   private:
    Ptr<UnivParams> phi_;
  };

  // The state equations are:
  //  mu[t+1] = mu[t] + delta[t] + u[t]
  //  delta[t+1] = D + phi * (delta[t]-D) + v[t]
  // To put this model in state space form requires a 3-dimensional state:
  //     alpha[t] = (mu[t], delta[t], D)
  // Here D is the time-invariant mean parameter of an AR1 model for
  // which phi is the AR1 coefficient.
  class LocalLinearTrendMeanRevertingSlopeStateModel
      : public StateModel,
        public CompositeParamPolicy,
        public IID_DataPolicy<VectorData>,
        public PriorPolicy
  {
   public:
    LocalLinearTrendMeanRevertingSlopeStateModel(
        Ptr<ZeroMeanGaussianModel> level,
        Ptr<NonzeroMeanAr1Model> slope);
    LocalLinearTrendMeanRevertingSlopeStateModel(
        const LocalLinearTrendMeanRevertingSlopeStateModel &rhs);
    virtual LocalLinearTrendMeanRevertingSlopeStateModel * clone()const;

    virtual void clear_data();

    virtual void observe_state(const ConstVectorView then,
                               const ConstVectorView now,
                               int time_now);
    virtual void observe_initial_state(const ConstVectorView &state);
    virtual uint state_dimension()const{return 3;}

    virtual void simulate_state_error(VectorView eta, int t)const;

    virtual Ptr<SparseMatrixBlock> state_transition_matrix(int t)const;
    virtual Ptr<SparseMatrixBlock> state_variance_matrix(int t)const;

    virtual SparseVector observation_matrix(int t)const;

    virtual Vec initial_state_mean()const;
    virtual Spd initial_state_variance()const;
    void set_initial_level_mean(double level_mean);
    void set_initial_level_sd(double level_sd);
    void set_initial_slope_mean(double slope_mean);
    void set_initial_slope_sd(double slope_sd);

    void simulate_initial_state(VectorView state)const;

   private:
    void check_dim(const ConstVectorView &)const;
    std::vector<Ptr<UnivParams> > get_variances();
    Ptr<ZeroMeanGaussianModel> level_;
    Ptr<NonzeroMeanAr1Model> slope_;

    SparseVector observation_matrix_;
    Ptr<LocalLinearTrendMeanRevertingSlopeMatrix> state_transition_matrix_;
    Ptr<UpperLeftDiagonalMatrix> state_variance_matrix_;
    double initial_level_mean_;
    double initial_slope_mean_;
    Spd initial_state_variance_;
  };


}
#endif // BOOM_LOCAL_LINEAR_TREND_MEAN_REVERTING_SLOPE_STATE_MODEL_HPP_
