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
#ifndef BOOM_STATE_SPACE_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_POSTERIOR_SAMPLER_HPP_

#include <Models/StateSpace/StateSpaceModelBase.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM{
  class StateSpacePosteriorSampler : public PosteriorSampler {
   public:
    StateSpacePosteriorSampler(StateSpaceModelBase *model,
                               RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   protected:
    // Samplers for models with observation equations that are
    // conditionally normal can override this function to impute the
    // mixing distribution.  For Gaussian observation models it is a
    // no-op.
    virtual void impute_nonstate_latent_data() {}

   private:
    StateSpaceModelBase *m_;
    bool latent_data_initialized_;
  };
}
#endif //BOOM_STATE_SPACE_POSTERIOR_SAMPLER_HPP_
