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

#include <Models/Mixtures/PosteriorSamplers/DirichletProcessMvnCollapsedGibbsSampler.hpp>
#include <distributions.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {
  namespace {
    typedef DirichletProcessMvnCollapsedGibbsSampler DPMCGS;

    // Parameters of the normal inverse Wishart model for (mu,
    // Siginv), where Siginv is the matrix inverse of the variance
    // matrix Sigma.  The model is
    //   (mu | Sigma) ~ N(mu0, Sigma / kappa)
    //         Sigma  ~ W(nu, sum_of_squares)
    //
    // Here mean refers to mu0, mean_sample_size refers to kappa, and
    // variance_sample_size refers to nu.
    struct NormalInverseWishartParameters {
      NormalInverseWishartParameters(const MvnGivenSigma &mean_model,
                                     const WishartModel &precision_model)
          : sum_of_squares(precision_model.sumsq()),
            variance_sample_size(precision_model.nu()),
            mean_sample_size(mean_model.kappa()),
            mean(mean_model.mu()) {}

      SpdMatrix sum_of_squares;
      double variance_sample_size;
      double mean_sample_size;
      Vector mean;
    };

    // Updates the parameters of the Normal inverse Wishart model
    // given data summarized in suf.
    NormalInverseWishartParameters compute_mvn_posterior(
        const MvnGivenSigma &mean_model,
        const WishartModel &precision_model,
        const MvnSuf &suf) {
      NormalInverseWishartParameters ans(mean_model, precision_model);
      if (suf.n() > 0) {
        ans.variance_sample_size += suf.n();
        ans.mean_sample_size += suf.n();
        ans.sum_of_squares += suf.center_sumsq();
        double weight =
            precision_model.nu() * suf.n() / ans.variance_sample_size;
        ans.sum_of_squares.add_outer(suf.ybar() - mean_model.mu(), weight);
        double shrinkage = mean_model.kappa() / (mean_model.kappa() + suf.n());
        ans.mean *= shrinkage;
        ans.mean.axpy(suf.ybar(), 1 - shrinkage);
      }
      return ans;
    }

  }  // namespace

  DPMCGS::DirichletProcessMvnCollapsedGibbsSampler(
      DirichletProcessMvnModel *model,
      Ptr<MvnGivenSigma> mean_base_measure,
      Ptr<WishartModel> precision_base_measure,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_base_measure_(mean_base_measure),
        precision_base_measure_(precision_base_measure)
  {}

  double DPMCGS::logpri() const {
    report_error("Calling logpri for a Dirichlet process mixture really "
                 "doesn't make a lot of sense");
    return 0;
  }

  void DPMCGS::draw() {
    draw_cluster_membership_indicators();
    draw_parameters();
  }

  void DPMCGS::draw_cluster_membership_indicators() {
    const std::vector<Ptr<VectorData> > &data(model_->dat());
    if (cluster_indicators_.empty()) {
      // Begin with all data in a single cluster.
      cluster_indicators_.resize(data.size(), -1);
      for (int i = 0; i < data.size(); ++i) {
        assign_data_to_cluster(data[i]->value(), 0);
        cluster_indicators_[i] = 0;
      }
    }

    for (int i = 0; i < data.size(); ++i) {
      const Vector &y(data[i]->value());
      remove_data_from_cluster(y, cluster_indicators_[i]);
      cluster_indicators_[i] = -1;
      Vector prob = cluster_membership_probability(y);
      int cluster_number = rmulti_mt(rng(), prob);
      model_->assign_data_to_cluster(y, cluster_number);
      cluster_indicators_[i] = cluster_number;
    }
  }

  void DPMCGS::draw_parameters() {
    for (int i = 0; i < model_->number_of_clusters(); ++i) {
      NormalInverseWishartParameters params = compute_mvn_posterior(
          *mean_base_measure_,
          *precision_base_measure_,
          *model_->cluster(i).suf());
      SpdMatrix Siginv = rWish_mt(rng(),
                                  params.variance_sample_size,
                                  params.sum_of_squares.inv());
      Vector mu = rmvn_ivar_mt(rng(),
                               params.mean,
                               params.mean_sample_size * Siginv);
      model_->set_component_params(i, mu, Siginv);
    }
  }

  Vector DPMCGS::cluster_membership_probability(const Vector &y) {
    Vector ans(model_->number_of_clusters() + 1);
    int n = model_->dat().size();
    for (int i = 0; i < model_->number_of_clusters(); ++i) {
      const MvnSuf &suf(*model_->cluster(i).suf());
      ans[i] = log(suf.n()) - log(n - 1 + model_->alpha())
          + log_marginal_density(y, suf);
    }
    ans.back() = log(model_->alpha()) - log(n - 1 + model_->alpha())
        + log_marginal_density(y, empty_suf_);

    ans.normalize_logprob();
    return ans;
  }

  // For the math, see Murphy (Machine Learning: A probabilistic
  // perspective) page 161 (eq: 5.29).  We omit any factors that only
  // depend on the 'sample size' (which is always 1), or other
  // quantities that will be the same for all elements of the cluster
  // membership probability vector.  We also rely on heavy
  // cancellation in ratios of the multivariate gamma function.
  double DPMCGS::log_marginal_density(
      const Vector &y, const MvnSuf &suf) const {
    NormalInverseWishartParameters prior = compute_mvn_posterior(
        *mean_base_measure_,
        *precision_base_measure_,
        suf);
    MvnSuf posterior_suf(suf);
    posterior_suf.update_raw(y);
    NormalInverseWishartParameters posterior = compute_mvn_posterior(
        *mean_base_measure_,
        *precision_base_measure_,
        posterior_suf);
    int dim = prior.mean.size();
    double ans =
        0.5 * dim * log(prior.mean_sample_size / posterior.mean_sample_size)
        + 0.5 * prior.variance_sample_size
              * prior.sum_of_squares.logdet()
        - 0.5 * posterior.variance_sample_size
              * posterior.sum_of_squares.logdet()
        + lgamma((prior.variance_sample_size + 1) / 2.0)
        - lgamma((prior.variance_sample_size + 1 - dim) / 2.0);
    return ans;
  }

  void DPMCGS::assign_data_to_cluster(const Vector &y, int cluster) {
    model_->assign_data_to_cluster(y, cluster);
  }

  void DPMCGS::remove_data_from_cluster(const Vector &y, int cluster) {
    double empty = (model_->cluster(cluster).suf()->n() == 1);
    model_->remove_data_from_cluster(y, cluster);
    if (empty) {
      for (int i = 0; i < cluster_indicators_.size(); ++i) {
        if (cluster_indicators_[i] >= cluster) {
          --cluster_indicators_[i];
        }
      }
    }
  }

}  // namespace BOOM
