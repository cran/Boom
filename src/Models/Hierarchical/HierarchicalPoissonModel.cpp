/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include <Models/Hierarchical/HierarchicalPoissonModel.hpp>

namespace BOOM {

  HierarchicalPoissonData::HierarchicalPoissonData(
      double event_count, double exposure)
      : event_count_(event_count),
        exposure_(exposure)
  {}

  HierarchicalPoissonData * HierarchicalPoissonData::clone() const {
    return new HierarchicalPoissonData(*this);
  }

  ostream & HierarchicalPoissonData::display(ostream &out) const {
    out << event_count_ << " " << exposure_;
    return out;
  }

  HierarchicalPoissonModel::HierarchicalPoissonModel(
      double lambda_prior_guess,
      double lambda_prior_sample_size)
      : prior_(new GammaModel(
            lambda_prior_sample_size, lambda_prior_guess, 0))
  {
    initialize();
  }

  HierarchicalPoissonModel::HierarchicalPoissonModel(Ptr<GammaModel> prior)
      : prior_(prior)
  {
    initialize();
  }

  HierarchicalPoissonModel::HierarchicalPoissonModel(
      const HierarchicalPoissonModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        prior_(rhs.prior_->clone())
  {
    initialize();
    for (int i = 0; i < rhs.number_of_groups(); ++i) {
      add_data_level_model(rhs.data_model(i)->clone());
    }
  }

  HierarchicalPoissonModel * HierarchicalPoissonModel::clone() const {
    return new HierarchicalPoissonModel(*this);
  }

  void HierarchicalPoissonModel::add_data_level_model(
      Ptr<PoissonModel> data_level_model) {
    data_level_models_.push_back(data_level_model);
    ParamPolicy::add_model(data_level_model);
  }

  void HierarchicalPoissonModel::clear_data() {
    data_level_models_.clear();
    ParamPolicy::clear();
    initialize();
    prior_->clear_data();
  }

  void HierarchicalPoissonModel::clear_client_data() {
    prior_->clear_data();
    for (int i = 0; i < data_level_models_.size(); ++i) {
      data_level_models_[i]->clear_data();
    }
  }

  void HierarchicalPoissonModel::clear_methods() {
    prior_->clear_methods();
    for (int i = 0; i < data_level_models_.size(); ++i) {
      data_level_models_[i]->clear_methods();
    }
  }

  void HierarchicalPoissonModel::combine_data(const Model &rhs, bool) {
    const HierarchicalPoissonModel & rhs_model(
        dynamic_cast<const HierarchicalPoissonModel &>(rhs));
    for(int i = 0; i < rhs_model.number_of_groups(); ++i) {
      add_data_level_model(rhs_model.data_level_models_[i]);
    }
  }

  void HierarchicalPoissonModel::add_data(Ptr<Data> dp) {
    Ptr<HierarchicalPoissonData> data_point =
        dp.dcast<HierarchicalPoissonData>();
    double events = data_point->event_count();
    double exposure = data_point->exposure();
    double lambda_hat = 1;
    if (exposure > 0 && events > 0) {
      if (events > 0) {
        lambda_hat = events / exposure;
      } else {
        lambda_hat = 1.0 / exposure;
      }
    }
    NEW(PoissonModel, model)(lambda_hat);
    model->suf()->set(events, exposure);
    add_data_level_model(model);
  }

  int HierarchicalPoissonModel::number_of_groups() const {
    return data_level_models_.size();
  }

  PoissonModel * HierarchicalPoissonModel::data_model(int which_group) {
    return data_level_models_[which_group].get();
  }

  const PoissonModel * HierarchicalPoissonModel::data_model(
      int which_group) const {
    return data_level_models_[which_group].get();
  }

  GammaModel * HierarchicalPoissonModel::prior_model() {
    return prior_.get();
  }

  const GammaModel * HierarchicalPoissonModel::prior_model() const {
    return prior_.get();
  }

  double HierarchicalPoissonModel::prior_mean() const {
    return prior_->mean();
  }

  double HierarchicalPoissonModel::prior_sample_size() const {
    return prior_->alpha();
  }

  void HierarchicalPoissonModel::initialize() {
    ParamPolicy::add_model(prior_);
    for (int i = 0; i < data_level_models_.size(); ++i) {
      ParamPolicy::add_model(data_level_models_[i]);
    }
  }

}  // namespace BOOM
