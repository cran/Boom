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

#include <Models/StateSpace/PosteriorSamplers/StudentLocalLinearTrendPosteriorSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <distributions.hpp>
#include <distributions/trun_gamma.hpp>
#include <Samplers/ScalarSliceSampler.hpp>

namespace {
  // A local namespace for minor implementation details.
  class NuPosterior {
   public:
    NuPosterior(const BOOM::DoubleModel *nu_prior,
                const BOOM::GammaSuf *suf)
        : nu_prior_(nu_prior), suf_(suf) {}

    // Returns the un-normalized log posterior evaulated at nu.
    double operator()(double nu)const{
      double n = suf_->n();
      double sum = suf_->sum();
      double sumlog = suf_->sumlog();
      double nu2 = nu / 2.0;

      double ans = nu_prior_->logp(nu);
      ans += n * (nu2 * log(nu2) - lgamma(nu2));
      ans += (nu2 - 1) * sumlog;
      ans -= nu2 * sum;
      return ans;
    }
   private:
    const BOOM::DoubleModel *nu_prior_;
    const BOOM::GammaSuf *suf_;
  };

  inline double square(double x) {return x*x;}

  inline double draw_variance(const BOOM::WeightedGaussianSuf &suf,
                              const BOOM::GammaModelBase &prior,
                              double sigma_upper_limit,
                              BOOM::RNG &rng){
    // Prior is 1.0 / sigsq ~ Gamma(df/2, ss/2).
    // So df = 2 * alpha, and ss = 2 * beta.
    double df = suf.n() + 2 * prior.alpha();   // alpha = df / 2
    double ss = suf.sumsq() + 2 * prior.beta();

    double siginv;
    if (sigma_upper_limit == BOOM::infinity()) {
      siginv = BOOM::rgamma_mt(rng, df/2, ss/2);
    } else {
      siginv = BOOM::rtrun_gamma_mt(rng, df/2, ss/2,
                              1.0/square(sigma_upper_limit));
    }
    return 1.0 / siginv;
  }
}  // namespace

namespace BOOM {
  StudentLocalLinearTrendPosteriorSampler::StudentLocalLinearTrendPosteriorSampler(
      StudentLocalLinearTrendStateModel *model,
      Ptr<GammaModelBase> sigsq_level_prior,
      Ptr<DoubleModel> nu_level_prior,
      Ptr<GammaModelBase> sigsq_slope_prior,
      Ptr<DoubleModel> nu_slope_prior)
      : model_(model),
        sigsq_level_prior_(sigsq_level_prior),
        nu_level_prior_(nu_level_prior),
        sigsq_slope_prior_(sigsq_slope_prior),
        nu_slope_prior_(nu_slope_prior),
        sigma_level_upper_limit_(infinity()),
        sigma_slope_upper_limit_(infinity())
  {}

  double StudentLocalLinearTrendPosteriorSampler::logpri()const{
    return sigsq_level_prior_->logp(1.0 / model_->sigsq_level())
        + nu_level_prior_->logp(model_->nu_level())
        + sigsq_slope_prior_->logp(1.0 / model_->sigsq_slope())
        + nu_slope_prior_->logp(1.0 / model_->nu_slope());
  }

  void StudentLocalLinearTrendPosteriorSampler::draw(){
    draw_sigsq_level();
    draw_nu_level();
    draw_sigsq_slope();
    draw_nu_slope();
  }

  void StudentLocalLinearTrendPosteriorSampler::set_sigma_level_upper_limit(
      double upper_limit){
    sigma_level_upper_limit_ = upper_limit;
  }

  void StudentLocalLinearTrendPosteriorSampler::set_sigma_slope_upper_limit(
      double upper_limit){
    sigma_slope_upper_limit_ = upper_limit;
  }

  void StudentLocalLinearTrendPosteriorSampler::draw_sigsq_level(){
    double sigsq = draw_variance(model_->sigma_slope_complete_data_suf(),
                                 *sigsq_level_prior_,
                                 sigma_level_upper_limit_,
                                 rng());
    model_->set_sigsq_level(sigsq);
  }

  void StudentLocalLinearTrendPosteriorSampler::draw_sigsq_slope(){
    double sigsq = draw_variance(model_->sigma_slope_complete_data_suf(),
                                 *sigsq_slope_prior_,
                                 sigma_slope_upper_limit_,
                                 rng());
    model_->set_sigsq_slope(sigsq);
  }

  void StudentLocalLinearTrendPosteriorSampler::draw_nu_level(){
    NuPosterior logpost(nu_level_prior_.get(),
                        &model_->nu_level_complete_data_suf());
    ScalarSliceSampler sampler(logpost, true);
    sampler.set_lower_limit(0.0);
    double nu = sampler.draw(model_->nu_level());
    model_->set_nu_level(nu);
  }

  void StudentLocalLinearTrendPosteriorSampler::draw_nu_slope(){
    NuPosterior logpost(nu_slope_prior_.get(),
                        &model_->nu_slope_complete_data_suf());
    ScalarSliceSampler sampler(logpost, true);
    sampler.set_lower_limit(0.0);
    double nu = sampler.draw(model_->nu_slope());
    model_->set_nu_slope(nu);
  }

} // namespace BOOM
