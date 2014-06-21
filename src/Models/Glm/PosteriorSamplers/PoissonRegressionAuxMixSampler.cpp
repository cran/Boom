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

#include <Models/Glm/PosteriorSamplers/PoissonRegressionAuxMixSampler.hpp>
#include <Models/Glm/PosteriorSamplers/poisson_mixture_approximation_table.hpp>
#include <distributions.hpp>

#ifndef NO_BOOST_THREADS
#include <boost/thread/thread.hpp>
#endif

namespace BOOM {

  // This is the public constructor to be called by a master node.
  PoissonRegressionAuxMixSampler::PoissonRegressionAuxMixSampler(
      PoissonRegressionModel *model,
      Ptr<MvnBase> prior,
      int number_of_threads)
      : model_(model),
        prior_(prior),
        complete_data_suf_(model_->xdim()),
        data_imputer_(new PoissonDataImputer),
        first_time_(true),
        num_threads_(1),
        thread_id_(0)
  {
    // Note that num_threads_ is set to 1, and not the number of
    // threads actually used, because this is the constructor for the
    // master object.  num_threads_ needs to be set to 1 (and
    // thread_id = 0, for impute_latent_data method to work correctly
    // in the unthreaded case (i.e. when there is a master but no
    // workers).
#ifndef NO_BOOST_THREADS
    if (number_of_threads > 1) {
      workers_.reserve(number_of_threads);
      for (int i = 0; i < number_of_threads; ++i) {
        boost::shared_ptr<PoissonRegressionAuxMixSampler> data_imputer(
            new PoissonRegressionAuxMixSampler(
                model_,
                prior_,
                number_of_threads,
                i));
        workers_.push_back(data_imputer);
      }
    }
#endif
  }

  // This is the private constructor, to be used for building worker
  // nodes.
  PoissonRegressionAuxMixSampler::PoissonRegressionAuxMixSampler(
      PoissonRegressionModel *model,
      Ptr<MvnBase> prior,
      int num_threads,
      int thread_id)
      : model_(model),
        prior_(prior),
        complete_data_suf_(model->xdim()),
        data_imputer_(new PoissonDataImputer),
        first_time_(false),
        num_threads_(num_threads),
        thread_id_(thread_id)
  {}

  double PoissonRegressionAuxMixSampler::logpri()const{
    return prior_->logp(model_->Beta());
  }

  void PoissonRegressionAuxMixSampler::draw(){
    impute_latent_data();
    draw_beta_given_complete_data();
  }

  class PoissonRegressionDataImputer {
   public:
    PoissonRegressionDataImputer(PoissonRegressionAuxMixSampler *sampler)
        : sampler_(sampler) {}
    void operator()() { sampler_->impute_latent_data(); }
   private:
    PoissonRegressionAuxMixSampler *sampler_;
  };

  void PoissonRegressionAuxMixSampler::impute_latent_data_single_threaded(){
    const std::vector<Ptr<PoissonRegressionData> > &data(model_->dat());
    int n = data.size();
    for(int i = thread_id_; i < n; i += num_threads_){
      const PoissonRegressionData *dp = data[i].get();
      const Vec &x(dp->x());
      double eta = model_->predict(x);
      int y = dp->y();
      double exposure = dp->exposure();
      double internal_neglog_final_event_time;
      double internal_mu;
      double internal_weight;
      double neglog_final_interarrival_time;
      double external_mu;
      double external_weight;
      data_imputer_->impute(rng(), y, exposure, eta,
                            &internal_neglog_final_event_time,
                            &internal_mu,
                            &internal_weight,
                            &neglog_final_interarrival_time,
                            &external_mu,
                            &external_weight);
      if (y > 0) {
        complete_data_suf_.add_data(
            x, internal_neglog_final_event_time - internal_mu, internal_weight);
      }
      complete_data_suf_.add_data(
          x, neglog_final_interarrival_time - external_mu, external_weight);
    }
  }

  // The latent variable scheme imagines the event times of y[i]
  // events from a Poisson process that occur in the interval [0, 1].
  // The maximum of these events, denoted tau[i], is marginally
  // Gamma(y[i],1)/lambda[i], which means u[i] = -log(tau[i]) =
  // log(lambda[i]) + NegLogGamma(y[i], 1).  Given a draw of u[i] we
  // need to express NegLogGamma() as a mixture of normals.
  //
  // The conditional distribution of tau[i], given it is the largest
  // of y[i] events from a Poisson process, is the same as the maximum
  // of y[i] uniform random variables.  That is,
  //
  //       tau[i] | y[i] ~ Beta(y[i], 1)
  //
  // If y[i] = 0 then tau[i] = 0 as well, since no events occurred in
  // [0,1].  It is also necessary to account for the unused portion of
  // [0,1], which is done by sampling the first event after the end of
  // the interval.  The terminal event kappa[i] is marginally
  // exponential, and conditionally truncated exponential with support
  // above 1 - tau[i].
  void PoissonRegressionAuxMixSampler::impute_latent_data(){
    complete_data_suf_.clear();
#ifdef NO_BOOST_THREADS
    impute_latent_data_single_threaded();
    first_time_ = false;
#else
    if (first_time_ || workers_.empty()) {
      impute_latent_data_single_threaded();
      first_time_ = false;
    } else {
      // If this class is a master class using threads, and if at
      // least one trip through the data has taken place, then launch
      // a series of threads to do the data augmentation.
      std::vector<boost::shared_ptr<boost::thread> > threads;
      for (int i = 0; i < workers_.size(); ++i) {
        boost::shared_ptr<boost::thread> thread(
            new boost::thread(
                PoissonRegressionDataImputer(workers_[i].get())));
        threads.push_back(thread);
      }
      for (int i = 0; i < workers_.size(); ++i) {
        threads[i]->join();
        complete_data_suf_.combine(
            workers_[i]->complete_data_sufficient_statistics());
      }
    }
#endif
  }

  double PoissonRegressionAuxMixSampler::draw_final_event_time(int y){
    return rbeta_mt(rng(), y, 1);
  }

  void PoissonRegressionAuxMixSampler::draw_beta_given_complete_data(){
    Spd ivar = prior_->siginv() + complete_data_suf_.xtx();
    Vec ivar_mu = prior_->siginv() * prior_->mu() + complete_data_suf_.xty();
    Vec beta = rmvn_suf_mt(rng(), ivar, ivar_mu);
    model_->set_Beta(beta);
  }

  const WeightedRegSuf &
  PoissonRegressionAuxMixSampler::complete_data_sufficient_statistics()const{
    return complete_data_suf_;
  }

}  // namespace BOOM
