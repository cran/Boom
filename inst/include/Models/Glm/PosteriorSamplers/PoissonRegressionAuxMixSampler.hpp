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

#ifndef BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_
#define BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_

#include <memory>

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/Imputer.hpp>
#include <Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp>
#include <Models/Glm/PoissonRegressionModel.hpp>
#include <Models/Glm/WeightedRegressionModel.hpp>
#include <Models/MvnBase.hpp>
#include <Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp>

namespace BOOM{

  class PoissonDataImputer;

  class PoissonRegressionDataImputer
      : public LatentDataImputer<PoissonRegressionData, WeightedRegSuf> {
   public:
    // Args:
    //   coefficients: The coefficients for the model managed by the
    //     sampler.  These are constant for the duration of the data
    //     augmentation step, and then change (for all workers) after
    //     the parameter sampling step.
    PoissonRegressionDataImputer(const GlmCoefs *coefficients);

    virtual void impute_latent_data(
        const PoissonRegressionData &data_point,
        WeightedRegSuf *complete_data_suf,
        RNG &rng) const;

   private:
    const GlmCoefs *coefficients_;
    std::unique_ptr<PoissonDataImputer> imputer_;
  };

  //----------------------------------------------------------------------

  class PoissonRegressionAuxMixSampler
      : public PosteriorSampler {
   public:
    PoissonRegressionAuxMixSampler(PoissonRegressionModel *model,
                                   Ptr<MvnBase> prior,
                                   int number_of_threads = 1);

    virtual void draw();
    virtual double logpri()const;

    // Below this line are implementation details exposed for testing.
    void impute_latent_data();
    double draw_final_event_time(int y);
    double draw_censored_event_time(double final_event_time, double rate);
    double draw_censored_event_time_zero_case(double rate);

    void draw_beta_given_complete_data();
    const WeightedRegSuf &complete_data_sufficient_statistics()const;

    // Set the number of workers devoted to data augmentation, n >= 1.
    void set_number_of_workers(int n);

   private:
    PoissonRegressionModel *model_;
    Ptr<MvnBase> prior_;
    WeightedRegSuf complete_data_suf_;
    ParallelLatentDataImputer<PoissonRegressionData,
                              WeightedRegSuf,
                              PoissonRegressionModel> parallel_data_imputer_;
  };

}  // namespace BOOM

#endif // BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_
