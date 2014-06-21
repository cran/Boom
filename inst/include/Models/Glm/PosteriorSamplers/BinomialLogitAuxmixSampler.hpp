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
#include <Models/Glm/BinomialLogitModel.hpp>
#include <Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp>
#include <Models/MvnBase.hpp>

namespace BOOM {

  class BinomialLogitAuxmixSampler : public PosteriorSampler {
   public:
    enum ImputationMethod {Full, Partial};

    BinomialLogitAuxmixSampler(BinomialLogitModel *model,
                               Ptr<MvnBase> prior,
                               int clt_threshold = 10);
    virtual double logpri() const;
    virtual void draw();

    void impute_latent_data();
    void draw_params();

    void set_augmentation_method(ImputationMethod method,
                                 int clt_threshold = 10);

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
     private:
      mutable SpdMatrix xtx_;
      Vector xty_;
      mutable bool sym_;
    };

   protected:
    const SufficientStatistics & suf() const {return suf_;}

   private:
    BinomialLogitModel *model_;
    boost::shared_ptr<BinomialLogitDataImputer> data_imputer_;
    Ptr<MvnBase> prior_;
    SufficientStatistics suf_;
  };


}  // namespace BOOM

#endif  // BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_
