/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_STATE_SPACE_STUDENT_REGRESSION_MODEL_HPP_
#define BOOM_STATE_SPACE_STUDENT_REGRESSION_MODEL_HPP_

#include <Models/StateSpace/StateSpaceNormalMixture.hpp>
#include <Models/Glm/TRegression.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM {
  namespace StateSpace {
    class VarianceAugmentedRegressionData
        : public RegressionData {
     public:
      VarianceAugmentedRegressionData(double y, const Vector &x);
      double weight() const override {return weight_;}
      void set_weight(double weight);
      double offset() const {return offset_;}
      void set_offset(double offset);

     private:
      double weight_;
      double offset_;
    };
  }  // namespace StateSpace

  class StateSpaceStudentRegressionModel
      : public StateSpaceNormalMixture,
        public IID_DataPolicy<StateSpace::VarianceAugmentedRegressionData>,
        public PriorPolicy
  {
   public:
    StateSpaceStudentRegressionModel(int xdim);
    StateSpaceStudentRegressionModel(
        const Vector &y,
        const Matrix &X,
        const std::vector<bool> &observed = std::vector<bool>());
    StateSpaceStudentRegressionModel(
        const StateSpaceStudentRegressionModel &rhs);
    StateSpaceStudentRegressionModel * clone() const override;

    int time_dimension() const override;

    const StateSpace::VarianceAugmentedRegressionData &
    data(int t) const override {
      return *(dat()[t]);
    }

    // Returns the imputed observation variance from the latent data
    // for observation t.  This is sigsq() / w[t] from the comment
    // above.
    double observation_variance(int t) const override;

    // Returns the value for observation t minus x[t]*beta.  Returns
    // infinity if observation t is missing.
    double adjusted_observation(int t) const override;

    // Returns true if observation t is missing, false otherwise.
    bool is_missing_observation(int t) const override;

    TRegressionModel *observation_model() override {
      return observation_model_.get(); }
    const TRegressionModel *observation_model() const override {
      return observation_model_.get(); }

    // Set the offset in the data to the state contribution.
    void observe_data_given_state(int t) override;

    Vector simulate_forecast(const Matrix &predictors,
                             const Vector &final_state);
    Vector one_step_holdout_prediction_errors(
        RNG &rng,
        const Vector &response,
        const Matrix &predictors,
        const Vector &final_state);

   private:
    void initialize_param_policy();
    Ptr<TRegressionModel> observation_model_;
  };


}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STUDENT_REGRESSION_MODEL_HPP_
