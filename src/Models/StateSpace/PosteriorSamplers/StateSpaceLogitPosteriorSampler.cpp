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
#include <Models/StateSpace/PosteriorSamplers/StateSpaceLogitPosteriorSampler.hpp>
#include <distributions.hpp>

namespace BOOM {

  class StateSpaceLogitPosteriorSampler;
  namespace StateSpace {

    class LogitSufstatManager
        : public SufstatManagerBase {
     public:
      LogitSufstatManager(StateSpaceLogitPosteriorSampler *sampler)
          : sampler_(sampler) {}
      void clear_complete_data_sufficient_statistics() override {
        sampler_->clear_complete_data_sufficient_statistics(); }
      void update_complete_data_sufficient_statistics(int t) override {
        sampler_->update_complete_data_sufficient_statistics(t); }
     private:
      StateSpaceLogitPosteriorSampler *sampler_;
    };

  }  // namespace StateSpace

  namespace {
    typedef StateSpaceLogitPosteriorSampler SSLPS;
    typedef StateSpace::AugmentedBinomialRegressionData AugmentedData;
  }  // namespace

  SSLPS::StateSpaceLogitPosteriorSampler(
      StateSpaceLogitModel *model,
      Ptr<BinomialLogitSpikeSlabSampler> observation_model_sampler,
      RNG &seeding_rng)
      : StateSpacePosteriorSampler(model, seeding_rng),
        model_(model),
        observation_model_sampler_(observation_model_sampler),
        data_imputer_(observation_model_sampler->clt_threshold())
  {
    model_->register_data_observer(new StateSpace::LogitSufstatManager(this));
    observation_model_sampler_->fix_latent_data(true);
  }

  void SSLPS::impute_nonstate_latent_data() {
    const std::vector<Ptr<AugmentedData> > &data(model_->dat());
    for (int t = 0; t < data.size(); ++t) {
      Ptr<AugmentedData> dp = data[t];
      double precision_weighted_sum = 0;
      double total_precision = 0;
      double state_contribution = model_->observation_matrix(t).dot(
          model_->state(t));
      double regression_contribution = model_->observation_model()->predict(
          dp->x());
      std::tie(precision_weighted_sum, total_precision) = data_imputer_.impute(
          rng(),
          dp->n(),
          dp->y(),
          state_contribution + regression_contribution);
      dp->set_latent_data(precision_weighted_sum / total_precision,
                          1.0 / total_precision);
    }
  }

  void SSLPS::clear_complete_data_sufficient_statistics() {
    observation_model_sampler_->clear_complete_data_sufficient_statistics();
  }

  void SSLPS::update_complete_data_sufficient_statistics(int t) {
    Ptr<AugmentedData> dp = model_->dat()[t];
    double precision_weighted_mean = dp->latent_data_value();
    precision_weighted_mean -= dp->offset();
    double precision = 1.0 / dp -> latent_data_variance();
    observation_model_sampler_->update_complete_data_sufficient_statistics(
        precision_weighted_mean * precision, precision, dp->x());
  }


}   // namespace BOOM
