// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_ZERO_INFLATED_POISSON_SAMPLER_HPP_
#define BOOM_ZERO_INFLATED_POISSON_SAMPLER_HPP_

#include "Models/BetaModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/ZeroInflatedPoissonModel.hpp"

namespace BOOM {
  class ZeroInflatedPoissonSampler : public PosteriorSampler {
   public:
    ZeroInflatedPoissonSampler(ZeroInflatedPoissonModel *model,
                               const Ptr<GammaModel> &lambda_prior,
                               const Ptr<BetaModel> &zero_probability_prior,
                               RNG &seeding_rng = GlobalRng::rng);
    ZeroInflatedPoissonSampler *clone_to_new_host(
        Model *new_host) const override;
    void draw() override;
    double logpri() const override;

   private:
    ZeroInflatedPoissonModel *model_;
    Ptr<GammaModel> lambda_prior_;
    Ptr<BetaModel> zero_probability_prior_;
  };
}  // namespace BOOM
#endif  // BOOM_ZERO_INFLATED_POISSON_SAMPLER_HPP_
