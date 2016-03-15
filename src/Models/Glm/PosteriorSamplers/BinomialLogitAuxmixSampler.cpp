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
    typedef BinomialLogisticRegressionDataImputer BLRDI;
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

  void BLAMS::SufficientStatistics::combine(
      const BLAMS::SufficientStatistics &rhs) {
    xtx_ += rhs.xtx_;
    xty_ += rhs.xty_;
    sym_ = sym_ && rhs.sym_;
  }

  BLAMS::BinomialLogitAuxmixSampler(BinomialLogitModel *model,
                                    Ptr<MvnBase> prior,
                                    int clt_threshold,
                                    RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior),
        suf_(model->xdim()),
        clt_threshold_(clt_threshold),
        parallel_data_imputer_(suf_, model_),
        latent_data_fixed_(false)
  {
    set_number_of_workers(1);
  }

  double BLAMS::logpri() const {
    return prior_->logp(model_->Beta());
  }

  void BLAMS::draw() {
    impute_latent_data();
    draw_params();
  }

  void BLAMS::impute_latent_data() {
    if (!latent_data_fixed_) {
      suf_ = parallel_data_imputer_.impute();
    }
  }

  void BLAMS::set_number_of_workers(int n) {
    if (n < 1) {
      report_error("At least one data impute worker is needed.");
    }
    parallel_data_imputer_.clear_workers();
    for (int i = 0; i < n; ++i) {
      parallel_data_imputer_.add_worker(
          new BinomialLogisticRegressionDataImputer(
              clt_threshold_,
              model_->coef_prm().get()),
          rng());
    }
    parallel_data_imputer_.assign_data();
  }

  void BLAMS::draw_params() {
    SpdMatrix ivar = prior_->siginv() + suf_.xtx();
    Vector ivar_mu = suf_.xty() + prior_->siginv() * prior_->mu();
    Vector draw = rmvn_suf_mt(rng(), ivar, ivar_mu);
    model_->set_Beta(draw);
  }

  void BLAMS::fix_latent_data(bool fixed) {
    latent_data_fixed_ = fixed;
  }

  void BLAMS::clear_complete_data_sufficient_statistics() {
    suf_.clear();
  }

  void BLAMS::update_complete_data_sufficient_statistics(
      double precision_weighted_sum,
      double total_precision,
      const Vector &x) {
    suf_.update(x, precision_weighted_sum, total_precision);
  }

  //======================================================================

  BLRDI::BinomialLogisticRegressionDataImputer(int clt_threshold,
                                               const GlmCoefs *coef)
      : latent_data_imputer_(clt_threshold),
        coefficients_(coef)
  {}

  void BLRDI::impute_latent_data(
      const BinomialRegressionData &observation,
      Suf *suf,
      RNG &rng) const {
    const Vector &x(observation.x());
    double eta = coefficients_->predict(x);
    double sum, weight;
    try {
      std::pair<double, double> imputed = latent_data_imputer_.impute(
          rng,
          observation.n(),
          observation.y(),
          eta);
      sum = imputed.first;
      weight = imputed.second;
      suf->update(x, sum, weight);
    } catch(std::exception &e) {
      ostringstream err;
      err << "caught an exception "
          << "with the following message:"
          << e.what() << endl
          << "n   = " << observation.n() << endl
          << "y   = " << observation.y() << endl
          << "eta = " << eta << endl;
      report_error(err.str());
    }
  }


}  // namespace BOOM
