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

#include <distributions.hpp>
#include <Models/Glm/PosteriorSamplers/BinomialLogitAuxmixSampler.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {
  namespace {
    typedef BinomialLogitAuxmixSampler BLAMS;
  }  // namespace

  BLAMS::SufficientStatistics::SufficientStatistics(int dim)
      : xtx_(dim),
        xty_(dim),
        sym_(false)
  {}

  const SpdMatrix & BLAMS::SufficientStatistics::xtx() const {
    if (!sym_) {
      xtx_.reflect();
      sym_ = true;
    }
    return xtx_;
  }

  const Vector & BLAMS::SufficientStatistics::xty() const {
    return xty_;
  }

  void BLAMS::SufficientStatistics::update(
      const Vector &x, double weighted_value, double weight) {
    sym_ = false;
    xtx_.add_outer(x, weight, false);
    xty_.axpy(x, weighted_value);
  }

  void BLAMS::SufficientStatistics::clear() {
    xtx_ = 0;
    xty_ = 0;
    sym_ = false;
  }

  BLAMS::BinomialLogitAuxmixSampler(BinomialLogitModel *model,
                                    Ptr<MvnBase> prior,
                                    int clt_threshold)
      : model_(model),
        data_imputer_(new BinomialLogitCltDataImputer(clt_threshold)),
        prior_(prior),
        suf_(model->xdim())
  {}

  double BLAMS::logpri() const {
    return prior_->logp(model_->Beta());
  }

  void BLAMS::draw() {
    impute_latent_data();
    draw_params();
  }

  void BLAMS::impute_latent_data() {
    const std::vector<Ptr<BinomialRegressionData> > & data(model_->dat());
    suf_.clear();
    for (int i = 0; i < data.size(); ++i) {
      const BinomialRegressionData &observation(*(data[i]));
      const Vector &x(observation.x());
      double eta = model_->predict(x);
      double sum, weight;
      try {
        std::pair<double, double> imputed =
            data_imputer_->impute(rng(), observation.n(), observation.y(), eta);
        sum = imputed.first;
        weight = imputed.second;
        suf_.update(x, sum, weight);
      } catch(std::exception &e) {
        ostringstream err;
        err << "BinomialLogitAuxmixSampler::impute_latent_data "
            << "caught an exception "
            << "with the following message:"
            << e.what() << endl
            << "n   = " << observation.n() << endl
            << "y   = " << observation.y() << endl
            << "eta = " << eta << endl;
        report_error(err.str());
      }
    }
  }

  void BLAMS::draw_params() {
    Spd ivar = prior_->siginv() + suf_.xtx();
    Vector ivar_mu = suf_.xty() + prior_->siginv() * prior_->mu();
    Vector draw = rmvn_suf_mt(rng(), ivar, ivar_mu);
    model_->set_Beta(draw);
  }

  void BLAMS::set_augmentation_method(ImputationMethod method,
                                      int clt_threshold) {
    if (method == Full) {
      data_imputer_.reset(
          new BinomialLogitCltDataImputer(clt_threshold));
    } else if (method == Partial) {
      data_imputer_.reset(
          new BinomialLogitPartialAugmentationDataImputer(clt_threshold));
    }
  }

}  // namespace BOOM
