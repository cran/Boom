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
#include <Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <distributions.hpp>
#include <boost/function.hpp>  // TODO(stevescott):  change to std::function
#include <boost/bind.hpp>      // TODO(stevescott):  change to std::lambda

namespace BOOM {

  namespace {
  typedef SpikeSlabDaRegressionSampler SSDRS;
  }

  SSDRS::SpikeSlabDaRegressionSampler(
      RegressionModel *model,
      Ptr<IndependentMvnModelGivenScalarSigma> beta_prior,
      Ptr<GammaModelBase> siginv_prior,
      const Vector &prior_inclusion_probabilities)
      : model_(model),
        beta_prior_(beta_prior),
        siginv_prior_(siginv_prior),
        log_prior_inclusion_probabilities_(
            prior_inclusion_probabilities.size()),
        log_prior_exclusion_probabilities_(
            prior_inclusion_probabilities.size()),
        missing_design_matrix_(model_->xdim(), model_->xdim()),
        missing_y_(model_->xdim()),
        complete_data_xtx_diagonal_(model_->xdim()),
        complete_data_xty_(model_->xdim()),
        complete_data_yty_(0.0),
        prior_is_current_(false)
  {
    for (int i = 0; i < log_prior_inclusion_probabilities_.size(); ++i) {
      double p = prior_inclusion_probabilities[i];
      log_prior_inclusion_probabilities_[i] =
          p > 0 ? log(p): negative_infinity();
      p = 1.0 - p;
      log_prior_exclusion_probabilities_[i] =
          p > 0 ? log(p) : negative_infinity();
    }
    determine_missing_design_matrix();
    beta_prior_->prm1()->add_observer(
        boost::bind(&SSDRS::observe_changes_in_prior, this));
    beta_prior_->prm2()->add_observer(
        boost::bind(&SSDRS::observe_changes_in_prior, this));
    check_prior();
  }

  double SSDRS::logpri() const {
    check_prior();
    const Selector &g(model_->coef().inc());
    double sigsq = model_->sigsq();
    double ans = siginv_prior_->logp(1.0 / sigsq);
    const Vector &beta(model_->Beta());
    for (int i = 0; i < log_prior_inclusion_probabilities_.size(); ++i) {
      if (g[i]) {
        ans += log_prior_inclusion_probabilities_[i]
            + dnorm(beta[i],
                    beta_prior_->mu()[i],
                    sigsq / unscaled_prior_precision_[i],
                    true);
      } else {
        ans += log_prior_exclusion_probabilities_[i];
      }
      if (ans <= BOOM::negative_infinity()) return ans;
    }
    return ans;
  }

  void SSDRS::draw() {
    impute_latent_data();
    draw_model_indicators();
    draw_sigma_given_complete_data();
    draw_beta_given_complete_data();
  }

  void SSDRS::impute_latent_data() {
    missing_y_.resize(missing_design_matrix_.nrow());
    complete_data_xty_ = model_->suf()->xty();
    complete_data_yty_ = model_->suf()->yty();
    model_->coef().predict(missing_design_matrix_, VectorView(missing_y_));
    double sigma = model_->sigma();
    for (int i = 0; i < missing_y_.size(); ++i) {
      missing_y_[i] += rnorm_mt(rng(), 0, sigma);
      complete_data_xty_.axpy(missing_design_matrix_.row(i), missing_y_[i]);
      complete_data_yty_ += square(missing_y_[i]);
    }
  }

  void SSDRS::draw_model_indicators() {
    Selector gamma = model_->coef().inc();
    int N = gamma.nvars_possible();
    for (int i = 0; i < N; ++i) {
      double inclusion_probability = compute_inclusion_probability(i);
      double u = runif_mt(rng());
      if (u < inclusion_probability) {
        gamma.add(i);
      } else {
        gamma.drop(i);
      }
    }
    model_->coef().set_inc(gamma);
  }

  // Returns the probability that variable j is included in the model,
  // given complete data and sigma, but integrating out beta.  The
  // formula for this is
  //
  //  pi[j] = unknown_constant * prior_probability[j]
  //    * (prior_precision[j] / posterior_precision[j])^.5
  //    * exp(-.5 SSE[j] + SSB[j])
  //
  // where
  // SSE[j] = -2*\tilde\beta[j] * xty[j] + \tilde\beta[j]^2 * xtx[j]
  // SSB[j] = (\tilde\beta[j] - b[j])^2 / prior_variance[j]
  // \tilde\beta = xty[j] / (xtx[j] + prior_information[j])
  double SSDRS::compute_inclusion_probability(int j) const {
    check_prior();
    double prior_information = unscaled_prior_information(j);
    double posterior_information =
        complete_data_xtx_diagonal_[j] + prior_information;
    double posterior_mean = posterior_mean_beta_given_complete_data(j);

    double SSE = -2 * posterior_mean * complete_data_xty_[j]
        + square(posterior_mean) * complete_data_xtx_diagonal_[j];
    double prior_mean = beta_prior_->mu()[j];
    double SSB = square(posterior_mean - prior_mean) * prior_information;

    // TODO(stevescott):
    // Check that sigsq is needed here.
    double logp_in = log_prior_inclusion_probabilities_[j]
        + log(prior_information)
        - log(posterior_information)
        - .5 * (SSE + SSB) / model_->sigsq();

    double logp_out = log_prior_exclusion_probabilities_[j];

    double M = std::max<double>(logp_in, logp_out);
    logp_in = exp(logp_in - M);
    logp_out = exp(logp_out - M);
    double total = logp_in + logp_out;
    return logp_in / total;
  }

  double SSDRS::prior_ss()const{
    return siginv_prior_->beta() * 2;
  }

  double SSDRS::prior_df()const{
    return siginv_prior_->alpha() * 2;
  }

  double SSDRS::unscaled_prior_information(int j)const{
    check_prior();
    return unscaled_prior_precision_[j];
  }

  void SSDRS::draw_beta_given_complete_data() {
    const Selector &gamma(model_->coef().inc());
    int n = gamma.nvars();
    int N = gamma.nvars_possible();
    Vector beta(N, 0.0);
    double sigsq = model_->sigsq();
    for (int i = 0; i < n; ++i) {
      int I = gamma.indx(i);
      double unscaled_posterior_information =
          complete_data_xtx_diagonal_[I] + unscaled_prior_information(I);
      double posterior_variance = sigsq / unscaled_posterior_information;
      double posterior_mean = posterior_mean_beta_given_complete_data(I);
      beta[I] = rnorm_mt(rng(), posterior_mean, sqrt(posterior_variance));
    }
    model_->coef().set_Beta(beta);
  }


// Here is the math for the draw of sigma given either complete or
// observed data.  With complete data, the prior and the xtx matrix
// will both be diagonal.
/*
\documentclass{article}
\usepackage{amsmath}
\newcommand{\nc}{\newcommand}
\nc{\bx}{{\bf x}}
\nc{\bX}{{\bf X}}
\nc{\by}{{\bf y}}
\nc{\ominv}{\Omega^{-1}}
\begin{document}

\begin{equation*}
  \begin{split}
  p(1/\sigma^2 |\by) &= K p(\by|\beta, \sigma)
                          p(\beta|\sigma)
                          p(1/\sigma^2) / p(\beta|\sigma, \by) \\
&= \left( \frac{1}{\sigma^2}\right)^{n/2} \exp\left(
  -\frac{1}{2} SSE(\tilde\beta) / \sigma^2 \right) \\
& \qquad \times
\left(\frac{1}{\sigma^2}\right)^{K/2} |\ominv|^{-1/2}
\exp\left( (\tilde\beta - \beta_0)^T\ominv(\tilde\beta -
  \beta_0)/\sigma^2\right)\\
& \qquad \times \left(\frac{1}{\sigma^2}\right)^{\nu / 2}
\exp\left(-\frac{ss}{2}\frac{1}{\sigma^2}\right)\\
& \qquad \div \left(\frac{1}{\sigma^2}\right)^{K/2}
\exp\left(-\frac{1}{2} (\tilde \beta - \tilde\beta)^T\ominv(\tilde
  \beta - \tilde\beta) / \sigma^2\right)\\
&= \left(\frac{1}{\sigma^2}\right)^{\frac{n + \nu}{2}}
\exp\left(-\frac{1}{2}  \left[ SSE(\tilde\beta) + M(\tilde\beta,
    \beta_0, \ominv) + ss\right]/\sigma^2 \right)\\
\end{split}
\end{equation*}

\end{document}
*/
  void SSDRS::draw_sigma_given_complete_data() {
    double siginv = 0;
    double ss = prior_ss() + complete_data_yty_;
    double df = model_->suf()->n() + prior_df();
    const Vector &beta(model_->Beta());
    const Selector &gamma(model_->coef().inc());
    for (int i = 0; i < gamma.nvars(); ++i) {
      int I = gamma.indx(i);
      ss += square(beta[I]) * complete_data_xtx_diagonal_[I];
      ss -= 2 * beta[I] * complete_data_xty_[I];
    }
    siginv = rgamma(df / 2.0, ss / 2.0);
    model_->set_sigsq(1.0 / siginv);
  }

  // This is an 'observer' to be attached to the parameters of the prior
  // distribution, so we can be notified when they change.
  void SSDRS::observe_changes_in_prior()const {
    prior_is_current_ = false;
  }

  double SSDRS::posterior_mean_beta_given_complete_data(int j) const {
    double posterior_information = complete_data_xtx_diagonal_[j] +
        unscaled_prior_precision_[j];
    return (complete_data_xty_[j] + information_weighted_prior_mean(j))
        / posterior_information;
  }

  void SSDRS::check_prior() const {
    if (!prior_is_current_) {
      unscaled_prior_precision_ =
          1.0 / beta_prior_->unscaled_variance_diagonal();
      information_weighted_prior_mean_ =
          beta_prior_->mu() * unscaled_prior_precision_;
    }
    prior_is_current_ = true;
  }

  double SSDRS::information_weighted_prior_mean(int j) const {
    check_prior();
    return information_weighted_prior_mean_[j];
  }

  void SSDRS::determine_missing_design_matrix() {
    Spd xtx = model_->suf()->xtx();
    int number_of_variables = ncol(xtx);
    Vector scale_factor(number_of_variables);
    scale_factor[0] = 1.0;
    double n = xtx(0, 0);
    for (int i = 1; i < number_of_variables; ++i) {
      double sum_of_squares = xtx(i, i);
      double mean = xtx(0, i) / n;
      double variance = (sum_of_squares - n * square(mean)) / (n - 1);
      if (variance <= 0) {
        variance = 1.0;
      }
      scale_factor[i] = sqrt(variance);
    }

    for (int i = 0; i < number_of_variables; ++i) {
      for (int j = 0; j < number_of_variables; ++j) {
        xtx(i, j) /= (scale_factor[i] * scale_factor[j]);
      }
    }

    complete_data_xtx_diagonal_ = largest_eigenvalue(xtx) * 1.1;
    Spd xtx_missing = -xtx;
    xtx_missing.diag() += complete_data_xtx_diagonal_;

    // Now xtx_missing is D - XTX_obs
    bool ok = true;
    int iteration_count = 0;
    const int max_number_of_iterations = 10;
    do {
      missing_design_matrix_ = xtx_missing.chol(ok).t();
      ++iteration_count;
      if (!ok) {
        xtx_missing.diag() += .1 * complete_data_xtx_diagonal_;
      }
    } while (!ok && iteration_count < max_number_of_iterations);
    if (!ok) {
      report_error("Cholesky decomposition failed in "
                   "SpikeSlabDaRegressionSampler::"
                   "determine_missing_design_matrix.");
    }

    for (int i = 0; i < number_of_variables; ++i) {
      missing_design_matrix_.col(i) *= scale_factor[i];
      complete_data_xtx_diagonal_[i] *= square(scale_factor[i]);
    }
  }
}
