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

#include <Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp>
#include <distributions.hpp>
#include <distributions/trun_logit.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {

  //======================================================================
  // A zero-mean scale mixture of normals approximation to the
  // logistic distribution.
  const NormalMixtureApproximation
  BinomialLogitDataImputer::mixture_approximation(
      Vector(9, 0),
      Vector("0.88437229872213 1.16097607474416 1.28021991084306 "
             "1.3592552924727 1.67589879794907 2.20287232043947 "
             "2.20507148325819 2.91944313615144 3.90807611741308"),
      Vector("0.038483985581272 0.13389889791451 0.0657842076622429 "
             "0.105680086433879 0.345939491553619 0.0442261124345564 "
             "0.193289780660134 0.068173066865908 0.00452437089387876"));

  //----------------------------------------------------------------------
  void BinomialLogitDataImputer::debug_status_message(
      ostream &out,
      int number_of_trials,
      int number_of_successes,
      double eta) const {
    out << "number_of_trials:    " << number_of_trials << endl
        << "number_of_successes: " << number_of_successes << endl
        << "eta:                 " << eta << endl;
  }

  BinomialLogitPartialAugmentationDataImputer::
  BinomialLogitPartialAugmentationDataImputer(int clt_threshold)
      : clt_threshold_(clt_threshold)
  {}

  std::pair<double, double> BinomialLogitPartialAugmentationDataImputer::impute(
      RNG &rng, int number_of_trials, int number_of_successes, double eta) {
    if (number_of_successes > number_of_trials) {
      ostringstream err;
      err << "The number of successes must not exceed the number of trials "
          << "in BinomialLogitPartialAugmentationDataImputer::impute()." << endl;
      debug_status_message(err, number_of_trials, number_of_successes, eta);
      report_error(err.str());
    }
    if (number_of_successes < 0 || number_of_trials < 0) {
      ostringstream err;
      err << "The number of successes and the number of trials must both "
          << "be non-negative in "
          << "BinomialLogitPartialAugmentationDataImputer::impute()." << endl;
      debug_status_message(err, number_of_trials, number_of_successes, eta);
      report_error(err.str());
    }
    const double pi_squared_over_3 = 3.289868133696452872;
    double information_weighted_sum = 0;
    double information = 0;
    if (number_of_trials < clt_threshold_) {
      for (int i = 0; i < number_of_trials; ++i) {
        bool success = i < number_of_successes;
        double latent_logit = rtrun_logit_mt(rng, eta, 0, success);
        // mu is unused because the mixture is a scale-mixture only,
        // but we need it for the API.
        double mu, sigsq;
        mixture_approximation.unmix(rng, latent_logit - eta, &mu, &sigsq);
        double current_weight = 1.0 / sigsq;
        information += current_weight;
        information_weighted_sum += latent_logit * current_weight;
      }
    } else {
      // Large sample case.  There are number_of_successes draws from
      // the positive side, and number_of_trials - number_of_successes
      // draws from the negative side.
      double mean_of_logit_sum = 0;
      double variance_of_logit_sum = 0;
      if (number_of_successes > 0) {
        mean_of_logit_sum += number_of_successes * trun_logit_mean(eta, 0, true);
        variance_of_logit_sum +=
            number_of_successes * trun_logit_variance(eta, 0, true);
      }
      int number_of_failures = number_of_trials - number_of_successes;
      if (number_of_failures > 0) {
        mean_of_logit_sum += number_of_failures * trun_logit_mean(eta, 0, false);
        variance_of_logit_sum +=
            number_of_failures * trun_logit_variance(eta, 0, false);
      }
      // The information_weighted_sum is the sum of the latent logits
      // (approximated by a normal), divided by the weight that each
      // term in the sum recieves (the variance of the logistic
      // distribution, pi^2/3).
      information_weighted_sum =
          rnorm_mt(rng, mean_of_logit_sum, sqrt(variance_of_logit_sum));
      information_weighted_sum /= pi_squared_over_3;

      // Each latent logit carries the same amount of information:
      // 1/pi_squared_over_3.
      information = number_of_trials /  pi_squared_over_3;
    }
    return std::make_pair(information_weighted_sum, information);
  }

  int BinomialLogitPartialAugmentationDataImputer::clt_threshold() const {
    return clt_threshold_;
  }

  //======================================================================

  BinomialLogitCltDataImputer::BinomialLogitCltDataImputer(int clt_threshold)
      : clt_threshold_(clt_threshold)
  {}

  std::pair<double, double> BinomialLogitCltDataImputer::impute(
      RNG & rng,
      int number_of_trials,
      int number_of_successes,
      double eta) {
    if (number_of_trials > clt_threshold()) {
      return impute_large_sample(rng, number_of_trials, number_of_successes, eta);
    } else {
      return impute_small_sample(rng, number_of_trials, number_of_successes, eta);
    }
  }

  std::pair<double, double> BinomialLogitCltDataImputer::impute_small_sample(
      RNG & rng, int number_of_trials, int number_of_successes, double eta) {
    double information_weighted_sum = 0;
    double information = 0;
    for (int i = 0; i < number_of_trials; ++i) {
      bool success = i < number_of_successes;
      double latent_logit = rtrun_logit_mt(rng, eta, 0, success);
      // mu is unused because the mixture is a scale-mixture only,
      // but we need it for the API.
      double mu, sigsq;
      mixture_approximation.unmix(rng, latent_logit - eta, &mu, &sigsq);
      double current_weight = 1.0 / sigsq;
      information += current_weight;
      information_weighted_sum += latent_logit * current_weight;
    }
    return std::make_pair(information_weighted_sum, information);
  }

  //----------------------------------------------------------------------
  std::pair<double, double> BinomialLogitCltDataImputer::impute_large_sample(
      RNG &rng,
      int number_of_trials,
      int number_of_successes,
      double eta) {
    double information = 0.0;
    const Vector & mixing_weights(mixture_approximation.weights());
    const Vector & sigma(mixture_approximation.sigma());
    double negative_logit_support = plogis(0, eta, 1, true);
    double positive_logit_support = plogis(0, eta, 1, false);
    p0_ = mixing_weights / negative_logit_support;
    p1_ = mixing_weights / positive_logit_support;
    for (int m = 0; m < mixture_approximation.dim(); ++m) {
      p0_[m] *= pnorm(0, eta, sigma[m], true);
      p1_[m] *= pnorm(0, eta, sigma[m], false);
    }

    // p0 is the probability distribution over the mixture component
    // indicators for the failures.  N0_ is the count of the number of
    // failures belonging to each mixture component.
    rmultinom_mt(rng, number_of_trials - number_of_successes, p0_/sum(p0_), N0_);

    // p1 is the probability distribution over the mixture component
    // indicators for the successes.  N1_ is the count of the number
    // of successes in each mixture component.
    rmultinom_mt(rng, number_of_successes, p1_/sum(p1_), N1_);

    double simulation_mean = 0;
    double simulation_variance = 0;
    for (int m = 0; m < N0_.size(); ++m) {
      int total_obs = N0_[m] + N1_[m];
      if (total_obs == 0) {
        continue;
      }
      double sigsq = square(sigma[m]);
      double sig4 = square(sigsq);
      information += total_obs / sigsq;
      double truncated_normal_mean;
      double truncated_normal_variance;
      double cutpoint = 0;
      if (N0_[m] > 0) {
        trun_norm_moments(eta, sigma[m],
                          cutpoint, false,
                          &truncated_normal_mean, &truncated_normal_variance);
        simulation_mean += N0_[m] * truncated_normal_mean / sigsq;
        simulation_variance += N0_[m] * truncated_normal_variance / sig4;
      }
      if (N1_[m] > 0) {
        trun_norm_moments(eta, sigma[m],
                          cutpoint, true,
                          &truncated_normal_mean, &truncated_normal_variance);
        simulation_mean += N1_[m] * truncated_normal_mean / sigsq;
        simulation_variance += N1_[m] * truncated_normal_variance / sig4;
      }
    }
    double information_weighted_sum =
        rnorm_mt(rng, simulation_mean, sqrt(simulation_variance));
    return std::make_pair(information_weighted_sum, information);
  }

  //----------------------------------------------------------------------
  int BinomialLogitCltDataImputer::clt_threshold() const {
    return clt_threshold_;
  }
}  // namespace BOOM
