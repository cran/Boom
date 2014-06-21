/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp>
#include <distributions.hpp>
#include <stats/moments.hpp>

namespace BOOM {

  DynamicRegressionStateModel::DynamicRegressionStateModel(const Matrix &X)
      : xdim_(ncol(X)),
        initial_state_mean_(ncol(X), 0.0),
        initial_state_variance_(ncol(X), 1.0),
        transition_matrix_(new IdentityMatrix(xdim_))
  {
    std::vector<Ptr<UnivParams> > diagonal_variances_;
    diagonal_variances_.reserve(xdim_);
    for (int i = 0; i < xdim_; ++i) {
      coefficient_transition_model_.push_back(new ZeroMeanGaussianModel);
      ParamPolicy::add_model(coefficient_transition_model_.back());
      diagonal_variances_.push_back(
          coefficient_transition_model_.back()->Sigsq_prm());
    }

    X_.reserve(nrow(X));
    for(int i = 0; i < nrow(X); ++i){
      SparseVector x(xdim_);
      for(int j = 0; j < xdim_; ++j) x[j] = X(i, j);
      X_.push_back(x);
    }

    predictor_variance_.reserve(ncol(X));
    for(int i = 0; i < ncol(X); ++i){
      predictor_variance_.push_back(var(X.col(i)));
    }

    transition_variance_matrix_.reset(new UpperLeftDiagonalMatrix(
        diagonal_variances_,
        diagonal_variances_.size()));
  }

  DynamicRegressionStateModel::DynamicRegressionStateModel(
      const DynamicRegressionStateModel &rhs)
      : StateModel(rhs),
        CompositeParamPolicy(rhs),
        NullDataPolicy(rhs),
        PriorPolicy(rhs),
        xdim_(rhs.xdim_),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_),
        X_(rhs.X_),
        transition_matrix_(rhs.transition_matrix_)
  {
    coefficient_transition_model_.reserve(xdim_);
    std::vector<Ptr<UnivParams> > diagonal_variances_;
    diagonal_variances_.reserve(xdim_);
    for(int i = 0; i < xdim_; ++i){
      coefficient_transition_model_.push_back(
          rhs.coefficient_transition_model_[i]->clone());
      add_model(coefficient_transition_model_.back());
      diagonal_variances_.push_back(
          coefficient_transition_model_.back()->Sigsq_prm());
    }

    transition_variance_matrix_.reset(new UpperLeftDiagonalMatrix(
        diagonal_variances_,
        diagonal_variances_.size()));
  }

  DynamicRegressionStateModel * DynamicRegressionStateModel::clone()const{
    return new DynamicRegressionStateModel(*this);}

  void DynamicRegressionStateModel::set_xnames(
      const std::vector<string> &xnames) {
    if (xnames.size() != state_dimension()) {
      std::ostringstream err;
      err << "Error in DynamicRegressionStateModel::set_xnames." << endl
          << "The size of xnames is " << xnames.size() << endl
          << "But ncol(X) is " << state_dimension() << endl;
      report_error(err.str());
    }
    xnames_ = xnames;
  }

  const std::vector<string> & DynamicRegressionStateModel::xnames()const{
    return xnames_;
  }

  void DynamicRegressionStateModel::clear_data() {
    for(int i = 0; i < coefficient_transition_model_.size(); ++i){
      coefficient_transition_model_[i]->clear_data();
    }
  }

  void DynamicRegressionStateModel::observe_state(
      const ConstVectorView then, const ConstVectorView now, int time_now){
    check_size(then.size());
    check_size(now.size());
    for(int i = 0; i < then.size(); ++i){
      double change = now[i] - then[i];
      coefficient_transition_model_[i]->suf()->update_raw(change);
    }
  }

  void DynamicRegressionStateModel::observe_initial_state(
      const ConstVectorView &state){}

  uint DynamicRegressionStateModel::state_dimension()const{return xdim_;}

  void DynamicRegressionStateModel::simulate_state_error(VectorView eta, int t)const{
    check_size(eta.size());
    for (int i = 0; i < eta.size(); ++i) {
      eta[i] = rnorm(0, coefficient_transition_model_[i]->sigma());
    }
  }

  Ptr<SparseMatrixBlock>
      DynamicRegressionStateModel::state_transition_matrix(int t)const{
    return transition_matrix_;
  }

  Ptr<SparseMatrixBlock>
      DynamicRegressionStateModel::state_variance_matrix(int t)const{
    return transition_variance_matrix_;
  }

  SparseVector DynamicRegressionStateModel::observation_matrix(int t)const{
    return X_[t];
  }

  Vector DynamicRegressionStateModel::initial_state_mean()const{
    return initial_state_mean_;
  }

  void DynamicRegressionStateModel::set_initial_state_mean(const Vector &mu){
    check_size(mu.size());
    initial_state_mean_ = mu;
  }

  Spd DynamicRegressionStateModel::initial_state_variance()const{
    return initial_state_variance_;
  }

  void DynamicRegressionStateModel::set_initial_state_variance(const Spd &V) {
    check_size(V.nrow());
    initial_state_variance_ = V;
  }

  const GaussianSuf * DynamicRegressionStateModel::suf(int i)const{
    return coefficient_transition_model_[i]->suf().get();
  }

  double DynamicRegressionStateModel::sigsq(int i)const{
    return coefficient_transition_model_[i]->sigsq();
  }

  void DynamicRegressionStateModel::set_sigsq(double sigsq, int i){
    coefficient_transition_model_[i]->set_sigsq(sigsq);
  }

  const Vector & DynamicRegressionStateModel::predictor_variance()const{
   return predictor_variance_;
  }

  Ptr<UnivParams> DynamicRegressionStateModel::Sigsq_prm(int i){
    return coefficient_transition_model_[i]->Sigsq_prm();
  }

  const Ptr<UnivParams> DynamicRegressionStateModel::Sigsq_prm(int i)const{
    return coefficient_transition_model_[i]->Sigsq_prm();
  }

  void DynamicRegressionStateModel::check_size(int n)const{
    if (n != xdim_) {
      report_error("Wrong sized vector or matrix argument in"
                   " DynamicRegressionStateModel");
    }
  }

}
