/*
  Copyright (C) 2014 Steven L. Scott

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

#ifndef BOOM_LATENT_DATA_IMPUTER_HPP
#define BOOM_LATENT_DATA_IMPUTER_HPP

#include <memory>
#include <cstddef>

#ifndef _WIN32
// Support for async/future is not yet available on the version of
// MinGW used by CRAN.
// TODO(stevescott): Remove the ugly conditional macros once CRAN can
// support this part of C++11.
#include <future>
#endif

#include <Models/ModelTypes.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {

  // A mix-in base class that provides impute_latent_data.
  //
  // Type requirements:
  //   OBSERVED_DATA: The type of observed data described by the raw
  //     model.  Should inherit from BOOM::Data (stored in a Ptr).
  //   SUFFICIENT_STATISTICS: The sufficient statistics for the
  //     augmented model.  The intent is that this class inherits from
  //     the Sufstat base class, but it is sufficient for it to have
  //     clear() and combine(const SUFFICIENT_STATISTICS &) methods.
  template <class OBSERVED_DATA,
            class SUFFICIENT_STATISTICS>
  class LatentDataImputer {
   public:
    // Impute the latent information associated with obs, and update suf.
    virtual void impute_latent_data(
        const OBSERVED_DATA &obs,
        SUFFICIENT_STATISTICS *complete_data_suf,
        RNG &random_number_generator) const = 0;

    virtual ~LatentDataImputer() {}
  };

  //======================================================================
  // This is a wrapper class that combines a LatentDataImputer with
  // other objects needed to work effectively as a member of a
  // ParallelLatentDataImputer worker pool:
  //   * An instance of complete data sufficient statistics.
  //   * A (thread safe) random number generator.
  //   * A vector of observed data to be augmented.
  template <class OBSERVED_DATA,
            class SUFFICIENT_STATISTICS>
  class LatentDataImputerWorker {
   public:
    typedef LatentDataImputer<OBSERVED_DATA, SUFFICIENT_STATISTICS> Imputer;

    LatentDataImputerWorker(Imputer *imputer,
                            const SUFFICIENT_STATISTICS &suf,
                            RNG &seeding_rng = GlobalRng::rng)
        : imputer_(imputer),
          suf_(suf)
    {
      forget_data();
      // If imputer also happens to be a PosteriorSampler, then use
      // the PosteriorSampler's methods for random number generation.
      PosteriorSampler *sampler = dynamic_cast<PosteriorSampler *>(imputer);
      if (sampler) {
        rng_ = &(sampler->rng());
      } else {
        rng_storage_.reset(new RNG(seed_rng(seeding_rng)));
        rng_ = rng_storage_.get();
      }
    }

    void forget_data() {
      observed_data_ = nullptr;
      first_data_point_ = 0;
      one_past_end_ = 0;
    }

    // Args:
    //   full_data: The complete vector of observed data.  The
    //     expectation is that this is owned by a Model.
    //   first: The position of the first data point in *full_data
    //     that this worker should impute.
    //   one_past_end: The position that is one past the last data
    //     point in *full_data that this worker should impute.
    //
    // The data will be forgotten (by a call to forget_data()) if
    // full_data is a nullptr, or first == one_past_end.
    void assign_data(const std::vector<Ptr<OBSERVED_DATA> > *full_data,
                     std::size_t first,
                     std::size_t one_past_end) {
      if (!full_data) {
        forget_data();
        return;
      }
      if (one_past_end > full_data->size()) {
        one_past_end = full_data->size();
      }
      if (one_past_end < first) {
        report_error("Last data point must come after first one.");
      }
      if (one_past_end == first) {
        // It would be nice to combine this with the first if
        // statement, but since one_past_end might change in the
        // second one, this check needs its own branch.
        forget_data();
        return;
      }
      observed_data_ = full_data;
      first_data_point_ = first;
      one_past_end_ = one_past_end;
    }

    // Imputes the latent data for the observed data that have been
    // assigned through set_data.
    //
    // TODO(stevescott): Note, the return value from this function is
    // not used.  It was set to bool after compiler errors related to
    // std::future<void> on CRAN's winbuilder platform.  Replace with
    // void in the future when possible.
    bool impute() {
      suf_.clear();

      // If there are more workers than data points then some workers
      // may not have data assigned.
      if (!observed_data_) {
        return true;
      }
      // Some syntactic sugar to prevent excessive *'s and ('s.
      const auto &data(*observed_data_);
      for (int i = first_data_point_; i < one_past_end_; ++i) {
        imputer_->impute_latent_data(*data[i], &suf_, rng());
      }
      return true;
    }

    const SUFFICIENT_STATISTICS &suf() const {
      return suf_;
    }

    RNG & rng() {
      return *rng_;
    }

    void set_seed(unsigned long seed) {
      rng_->seed(seed);
    }

    // The number of data points managed by this worker.
    std::size_t data_size() const {
      if (!observed_data_) {
        return 0;
      }
      return observed_data_->size();
    }

   private:
    std::unique_ptr<Imputer> imputer_;
    SUFFICIENT_STATISTICS suf_;

    // If the imputer happens to be a PosteriorSampler, then it brings
    // its own random number generator, in which case rng_ should
    // point to it.  If the imputer is not a PosteriorSampler then we
    // need to maintin our own RNG, which we do in rng_storage_.  If
    // rng_storage_ is needed, it will be pointed to by rng_.
    // rng_storage_ is never called directly.  It will only be called
    // through rng_.
    RNG *rng_;
    std::unique_ptr<RNG> rng_storage_;

    // The data managed by this worker.  This is a continguous
    // chunk of the data managed by the Model we're imputing data for.
    const std::vector<Ptr<OBSERVED_DATA> > *observed_data_;
    int first_data_point_;
    int one_past_end_;
  };

  //======================================================================

  // Implements a worker pool for drawing latent data in parallel.
  //
  // The idiom for using this class is
  // ParallelLatentDataImputer imputer(
  //     SpecificSufstats complete_data_suf,
  //     &a_specific_model);
  // for (int i = 0; i < number_of_workers; ++i) {
  //   imputer.add_worker(new DerivedImputer);
  // }
  // imputer.assign_data();
  // Sufstat = imputer.impute();
  template <class OBSERVED_DATA,
            class SUFFICIENT_STATISTICS,
            class MODEL>
  class ParallelLatentDataImputer {
   public:
    typedef LatentDataImputer<OBSERVED_DATA,
                              SUFFICIENT_STATISTICS> Imputer;
    typedef LatentDataImputerWorker<OBSERVED_DATA,
                                    SUFFICIENT_STATISTICS> Worker;
    // Args:
    //   suf: The complete data sufficient statistics to be imputed.
    //   model: A pointer to the model that is the source of the
    //     observed data.  The model should probably inherit from
    //     IID_DataPolicy<OBSERVED_DATA>.  It needs to provide a dat()
    //     method returning const std::vector<Ptr<OBSERVED_DATA>> &.
    ParallelLatentDataImputer(const SUFFICIENT_STATISTICS &suf,
                              MODEL *model)
        : suf_(suf),
          model_(model),
          first_pass_(true){}

    // Add a worker to the worker pool.  The intent is for each worker
    // to run in its own thread, though if there is only one worker no
    // multi-threading is attempted.
    //
    // Args:
    //   imputer: An imputer object.  This should be a freshly
    //     allocated object not shared with anyone else.  The worker
    //     pool will take ownership of the object and call delete when
    //     the ParallelLatentDataImputer is destroyed.
    void add_worker(Imputer *imputer, RNG &seeding_rng = GlobalRng::rng) {
      workers_.emplace_back(new Worker(imputer, suf_, seeding_rng));
    }

    int number_of_workers() const {
      return workers_.size();
    }

    void clear_workers() {
      workers_.clear();
    }

    // Impute the latent data (in parallel) and return the imputed
    // complete data sufficient statistics.
    const SUFFICIENT_STATISTICS & impute() {
      suf_.clear();
      if (workers_.empty()) {
        report_error("No workers have been assigned.");
      }
      // The data must be assigned each iteration for this to work
      // with mixtures, because each mixture component will have a
      // different number of observations each iteration.
      assign_data();
#ifdef _WIN32
      first_pass_ = true;
#endif
      if (first_pass_ || workers_.size() == 1) {
        // Because some tools (e.g.  eigen's blas) require an initial
        // run to initialize shared data, one pass is done without
        // threading before multiple threads are invoked.  Also, if
        // there is only one worker, take this path to avoid the
        // overhead of async/future.
        for (int i = 0; i < workers_.size(); ++i) {
          workers_[i]->impute();
          suf_.combine(workers_[i]->suf());
        }
        first_pass_ = false;
      } else {
#ifndef _WIN32
        std::vector<std::future<bool> > results;
        for (int i = 0; i < workers_.size(); ++i) {
          results.emplace_back(
              async(std::launch::async,
                    &Worker::impute,
                    workers_[i].get()));
        }
        for (int i = 0; i < workers_.size(); ++i) {
          results[i].get();
          suf_.combine(workers_[i]->suf());
        }
#endif
      }
      return suf_;
    }

    // Assigns the data controlled by model_ to the workers.
    void assign_data() {
      if (workers_.empty()) {
        report_error("assign_data called, but there are no workers.");
      }
      const std::vector<Ptr<OBSERVED_DATA>> &observed_data(model_->dat());
      std::size_t sample_size = observed_data.size();
      std::size_t chunk_size = sample_size / workers_.size();
      if (chunk_size * workers_.size() < sample_size) {
        ++chunk_size;
      }

      std::size_t b = 0;
      for(std::size_t i = 0; i < workers_.size(); ++i) {
        std::size_t e = std::min<std::size_t>(b + chunk_size, sample_size);
        workers_[i]->assign_data(&observed_data, b, e);
        b = e;
      }
    }

   private:
    SUFFICIENT_STATISTICS suf_;
    MODEL *model_;
    std::vector<std::unique_ptr<Worker> > workers_;
    bool first_pass_;
  };

}  // namespace BOOM

#endif // BOOM_LATENT_DATA_IMPUTER_HPP
