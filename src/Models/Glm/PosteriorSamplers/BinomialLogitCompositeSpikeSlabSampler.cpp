/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include <Models/Glm/PosteriorSamplers/BinomialLogitCompositeSpikeSlabSampler.hpp>
#include <distributions.hpp>
#include <Samplers/TIM.hpp>
#include <cpputil/math_utils.hpp>

#include <ctime>

namespace BOOM{
  double BinomialLogitLogPostChunk::operator()(const Vec &beta_chunk)const{
    Vec g;
    Mat h;
    return (*this)(beta_chunk, g, h, 0);
  }
  //----------------------------------------------------------------------
  double BinomialLogitLogPostChunk::operator()(
      const Vec &beta_chunk, Vec &grad, Mat &hess, int nd)const{
    Vec nonzero_beta = m_->included_coefficients();
    VectorView nonzero_beta_chunk(nonzero_beta, start_, chunk_size_);
    nonzero_beta_chunk = beta_chunk;

    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    const Selector &inc(m_->coef().inc());
    const Spd siginv(inc.select(pri_->siginv()));
    const Vec mu(inc.select(pri_->mu()));

    double ans = dmvn(nonzero_beta, mu, siginv, 0.0, true);
    if(nd > 0){
      Selector chunk_selector(nonzero_beta.size(), false);
      for(int i = start_; i < start_ + chunk_size_; ++i) chunk_selector.add(i);
      grad = -1*chunk_selector.select(siginv * (nonzero_beta - mu));
      if(nd > 1) {
        hess = chunk_selector.select(siginv);
        hess *= -1;
      }
    }

    int nobs = data.size();
    for(int i = 0; i < nobs; ++i){
      int yi = data[i]->y();
      int ni = data[i]->n();
      Vec x = inc.select(data[i]->x());
      double eta = nonzero_beta.dot(x);
      double prob = plogis(eta);
      ans += dbinom(yi, ni, prob, true);
      if(nd > 0){
        const ConstVectorView x_chunk(x, start_, chunk_size_);
        grad.axpy(x_chunk, yi - ni*prob);
        if(nd > 1){
          hess.add_outer(x_chunk, x_chunk, -ni * prob * (1-prob));
        }
      }
    }
    return ans;
  }
  //----------------------------------------------------------------------
  typedef BinomialLogitCompositeSpikeSlabSampler BLCSSS;
  BLCSSS::BinomialLogitCompositeSpikeSlabSampler(
      BinomialLogitModel *model,
      Ptr<MvnBase> prior,
      Ptr<VariableSelectionPrior> vpri,
      int clt_threshold,
      double tdf,
      int max_chunk_size,
      double rwm_variance_scale_factor)
      : BinomialLogitSpikeSlabSampler(model, prior, vpri, clt_threshold),
        m_(model),
        pri_(prior),
        tdf_(tdf),
        max_chunk_size_(max_chunk_size),
        rwm_variance_scale_factor_(rwm_variance_scale_factor),
        auxmix_tries_(0),
        auxmix_times_(0.0),
        rwm_tries_(0),
        rwm_times_(0.0),
        tim_tries_(0),
        tim_times_(0.0),
        rwm_chunk_attempts_(0),
        rwm_chunk_successes_(0),
        tim_chunk_attempts_(0),
        tim_chunk_mode_failures_(0),
        tim_chunk_successes_(0),
        rwm_chunk_times_(0.0),
        tim_mode_finding_times_(0.0),
        tim_trial_times_(0.0),
        tim_mode_finding_wasted_time_(0.0)
  {}
  //----------------------------------------------------------------------
  void BLCSSS::draw(){
    double u  = runif_mt(rng());
    clock_t start = clock();
    if(u < .333){
      ++auxmix_tries_;
      BinomialLogitSpikeSlabSampler::draw();
      clock_t end = clock();
      auxmix_times_ += double(end - start) / CLOCKS_PER_SEC;
    }else if(u < .6667){
      ++rwm_tries_;
      rwm_draw();
      clock_t end = clock();
      rwm_times_ += double(end - start) / CLOCKS_PER_SEC;
    }else{
      ++tim_tries_;
      tim_draw();
      clock_t end = clock();
      tim_times_ += double (end - start) / CLOCKS_PER_SEC;
    }
  }
  //----------------------------------------------------------------------
  void BLCSSS::rwm_draw(){
    if(m_->coef().nvars() == 0) return;
    int total_number_of_chunks = compute_number_of_chunks();
    for(int chunk = 0; chunk < total_number_of_chunks; ++chunk) {
      rwm_draw_chunk(chunk);
    }
  }
  //----------------------------------------------------------------------
  void BLCSSS::rwm_draw_chunk(int chunk){
    clock_t start = clock();
    const Selector &inc(m_->coef().inc());
    int nvars = inc.nvars();
    Vec full_nonzero_beta = m_->included_coefficients();   // only nonzero components
    // Compute information matrix for proposal distribution.  For
    // efficiency, also compute the log-posterior of the current beta.
    Vec mu(inc.select(pri_->mu()));
    Spd siginv(inc.select(pri_->siginv()));
    double original_logpost = dmvn(full_nonzero_beta, mu, siginv, 0, true);

    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    int nobs = data.size();

    int full_chunk_size = compute_chunk_size();
    int chunk_start = chunk * full_chunk_size;
    int elements_remaining = nvars - chunk_start;
    int this_chunk_size = std::min(elements_remaining, full_chunk_size);
    Selector chunk_selector(nvars, false);
    for(int i = chunk_start; i< chunk_start + this_chunk_size; ++i) {
      chunk_selector.add(i);
    }

    Spd proposal_ivar = chunk_selector.select(siginv);

    for(int i = 0; i < nobs; ++i){
      Vec x = inc.select(data[i]->x());
      double eta = x.dot(full_nonzero_beta);
      double prob = plogis(eta);
      double weight = prob * (1-prob);
      VectorView x_chunk(x, chunk_start, this_chunk_size);
      // Only upper triangle is accessed.  Need to reflect at end of loop.
      proposal_ivar.add_outer(x_chunk, weight, false);
      int yi = data[i]->y();
      int ni = data[i]->n();
      original_logpost += dbinom(yi, ni, prob, true);
    }
    proposal_ivar.reflect();
    VectorView beta_chunk(full_nonzero_beta, chunk_start, this_chunk_size);
    if(tdf_ > 0){
      beta_chunk = rmvt_ivar_mt(
          rng(), beta_chunk, proposal_ivar / rwm_variance_scale_factor_, tdf_);
    }else{
      beta_chunk = rmvn_ivar_mt(
          rng(), beta_chunk, proposal_ivar / rwm_variance_scale_factor_);
    }

    double logpost = dmvn(full_nonzero_beta, mu, siginv, 0, true);
    Vec full_beta(inc.expand(full_nonzero_beta));
    logpost += m_->log_likelihood(full_beta, 0, 0, false);
    double log_alpha = logpost - original_logpost;
    double logu = log(runif_mt(rng()));
    ++rwm_chunk_attempts_;
    if(logu < log_alpha){
      m_->set_included_coefficients(full_nonzero_beta);
      ++rwm_chunk_successes_;
    }
    clock_t end = clock();
    rwm_chunk_times_ += double(end - start) / CLOCKS_PER_SEC;
  }
  //----------------------------------------------------------------------
  void BLCSSS::tim_draw(){
    int nvars = m_->coef().nvars();
    if(nvars == 0) return;
    int chunk_size = compute_chunk_size();
    int number_of_chunks = compute_number_of_chunks();
    assert(number_of_chunks * chunk_size >= nvars);

    for(int chunk = 0; chunk < number_of_chunks; ++chunk) {
      ++tim_chunk_attempts_;
      clock_t mode_start = clock();
      TIM tim_sampler(log_posterior(chunk), tdf_);
      Vec beta = m_->included_coefficients();
      int start = chunk_size * chunk;
      int elements_remaining = nvars - start;
      VectorView beta_chunk(beta,
                            start,
                            std::min(elements_remaining, chunk_size));
      bool ok = tim_sampler.locate_mode(beta_chunk);
      clock_t mode_end = clock();
      double mode_time = double(mode_end - mode_start) / CLOCKS_PER_SEC;
      tim_mode_finding_times_ += mode_time;
      if(ok){
        tim_sampler.fix_mode(true);
        clock_t trial_start = clock();
        beta_chunk = tim_sampler.draw(beta_chunk);
        m_->set_included_coefficients(beta);
        tim_chunk_successes_ += tim_sampler.last_draw_was_accepted();
        clock_t trial_end = clock();
        tim_trial_times_ += double(trial_end - trial_start) / CLOCKS_PER_SEC;
      }else{
        rwm_draw_chunk(chunk);
        tim_mode_finding_wasted_time_ += mode_time;
        ++tim_chunk_mode_failures_;
      }
    }
  }
  //----------------------------------------------------------------------
  BinomialLogitLogPostChunk BLCSSS::log_posterior(int chunk)const{
    return BinomialLogitLogPostChunk(
        m_, pri_.get(), compute_chunk_size(), chunk);
  }
  //----------------------------------------------------------------------
  int BLCSSS::compute_chunk_size()const{
    int nvars = m_->coef().nvars();
    if(max_chunk_size_ <= 0) return nvars;
    int number_of_full_chunks = nvars / max_chunk_size_;
    bool has_partial_chunk = number_of_full_chunks * max_chunk_size_ < nvars;
    int total_chunks = number_of_full_chunks + has_partial_chunk;
    int full_chunk_size = divide_rounding_up(nvars, total_chunks);
    return full_chunk_size;
  }
  //----------------------------------------------------------------------
  int BLCSSS::compute_number_of_chunks()const{
    if(max_chunk_size_ <= 0) return 1;
    int nvars = m_->coef().nvars();
    int number_of_full_chunks = nvars / max_chunk_size_;
    bool has_partial_chunk = number_of_full_chunks * max_chunk_size_ < nvars;
    return number_of_full_chunks + has_partial_chunk;
  }

  ostream & BLCSSS::time_report(ostream &out)const{
    out << "auxmix:  " << auxmix_tries_ << " iterations in "
        << auxmix_times_ << " seconds, for "
        << auxmix_tries_ / auxmix_times_ << " iterations / sec." << endl
        << "rwm:     " << rwm_tries_ << " iterations int "
        << rwm_times_ << " secconds, for "
        << rwm_tries_ / rwm_times_ << " iterations / sec." << endl
        << "tim:     " << tim_tries_ << " iterations int "
        << tim_times_ << " secconds, for "
        << tim_tries_ / tim_times_ << " iterations / sec." << endl
        << "TIM failed to locate the mode " << tim_chunk_mode_failures_
        << " times out of " << tim_chunk_attempts_ << " for a failure rate of "
        << double(tim_chunk_mode_failures_) / tim_chunk_attempts_ << "." << endl
        << "TIM spent " << tim_mode_finding_times_
        << " seconds finding the posterior mode, of which "
        << tim_mode_finding_wasted_time_ << " seconds were wasted.  That means "
        << tim_mode_finding_wasted_time_ / tim_mode_finding_times_
        << " of the total mode finding time was wasted." << endl
        << "The proportion of proposals from TIM chunks that were accepted is "
        << double(tim_chunk_successes_) / tim_chunk_attempts_ << "." << endl
        << "The proportion of proposals from RWM chunks that were accepted is "
        << double(rwm_chunk_successes_) / rwm_chunk_attempts_ << "." << endl;
    return out;
  }
};
