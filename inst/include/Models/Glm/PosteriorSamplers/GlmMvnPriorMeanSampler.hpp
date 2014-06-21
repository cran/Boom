/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_GLM_MVN_PRIOR_MEAN_SAMPLER_HPP
#define BOOM_GLM_MVN_PRIOR_MEAN_SAMPLER_HPP
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/GlmMvnPrior.hpp>
#include <Models/MvnBase.hpp>

namespace BOOM{
  class GlmMvnPriorMeanSampler
    : public PosteriorSampler
  {
  public:
    GlmMvnPriorMeanSampler(GlmMvnPrior *Mod,
			   Ptr<MvnBase> Pri);
    void draw();
    double logpri()const;
  private:
    GlmMvnPrior *mod;
    Ptr<MvnBase> pri;
    Spd ivar;
    Vec mu;
  };
}
#endif// BOOM_MVN_PRIOR_MEAN_SAMPLER_HPP
