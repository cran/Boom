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

#ifndef BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_
#define BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/Imputer.hpp>

#include <Models/Glm/BinomialLogitModel.hpp>
#include <Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp>
#include <Models/MvnBase.hpp>

namespace BOOM {

  class BinomialLogitAuxmixSampler : public PosteriorSampler {
   public:
    enum ImputationMethod {Full, Partial};

    BinomialLogitAuxmixSampler(BinomialLogitModel *model,
                               Ptr<MvnBase> prior,
                               int clt_threshold = 10,
                               RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

    void impute_latent_data();
    void draw_params();

    // Use up to 'n' logical threads to simulate the latent data.  If
    // n<= 0 then run single threaded.
    void set_number_of_workers(int n);

    // A sufficient statistics class to hold the sufstats from the
    // sampling algorithm.
    class SufficientStatistics {
     public:
      SufficientStatistics(int dim);
      const SpdMatrix &xtx() const;
      const Vector &xty()const;
      void update(const Vector &x,
                  double weighted_value,
                  double weight);
      void clear();
      void combine(const SufficientStatistics &rhs);
     private:
      mutable SpdMatrix xtx_;
      Vector xty_;
      mutable bool sym_;
    };

    // By default, this class updates its own latent data through a
    // call to impute_latent_data().  Calling this function with a
    // 'true' argument (the default), sets a flag that turns
    // impute_latent_data into a no-op.  The latent data can still be
    // manipulated through calls to clear_sufficient_statistics() and
    // update_sufficient_statistics(), but implicit data augmentation
    // is turned off.  Calling this function with a 'false' argument
    // turns data augmentation back on.
    void fix_latent_data(bool fixed = true);

    // Clear the complete data sufficient statistics.  This is
    // normally unnecessary.  This function is primarily intended for
    // nonstandard situations where the complete data sufficient
    // statistics need to be manipulated by an outside actor.
    void clear_complete_data_sufficient_statistics();

    // Increment the complete data sufficient statistics with the
    // given quantities.  This is normally unnecessary.  This function
    // is primarily intended for nonstandard situations where the
    // complete data sufficient statistics need to be manipulated by
    // an outside actor.
    void update_complete_data_sufficient_statistics(
        double precision_weighted_sum,
        double total_precision,
        const Vector &x);

    int clt_threshold() const {return clt_threshold_;}

   protected:
    const SufficientStatistics & suf() const {return suf_;}

   private:
    BinomialLogitModel *model_;
    Ptr<MvnBase> prior_;
    SufficientStatistics suf_;
    int clt_threshold_;
    ParallelLatentDataImputer<BinomialRegressionData,
                              SufficientStatistics,
                              BinomialLogitModel> parallel_data_imputer_;

    // A flag that can be use to turn off data augmentation.  If this
    // flag is set then impute_latent_data is a no-op.
    bool latent_data_fixed_;
  };

  //======================================================================
  class BinomialLogisticRegressionDataImputer
      : public LatentDataImputer<
                   BinomialRegressionData,
                   BinomialLogitAuxmixSampler::SufficientStatistics> {
   public:
    typedef BinomialLogitAuxmixSampler::SufficientStatistics Suf;

    // Args:
    //   clt_threshold: The number of iterations 'n' at which one
    //    should trust the central limit theorem.  The complete data
    //    sufficient statistics are sums of conditionally normal
    //    random variables.  If the number of trials for an
    //    observation is less than 'clt_threshold' then a separate
    //    latent variable will be imputed for each trial.  If it is
    //    larger than clt_threshold then the moments of the sum will
    //    be computed and a normal approximation will be used instead.
    //    A small integer like 5 seems to work well here.
    //   coef: A pointer to a set of logisitic regression
    //     coefficients, owned by the model for which this imputer
    //     is responsible.
    BinomialLogisticRegressionDataImputer(
        int clt_threshold,
        const GlmCoefs* coef);

    virtual void impute_latent_data(
        const BinomialRegressionData &data,
        Suf *suf,
        RNG &rng) const;

   private:
    BinomialLogitCltDataImputer latent_data_imputer_;
    const GlmCoefs *coefficients_;
  };

}  // namespace BOOM

#endif  // BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_
