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

#ifndef BOOM_GLM_MVN_SIGINV_IND_SAMPLER_HPP
#define BOOM_GLM_MVN_SIGINV_IND_SAMPLER_HPP

#include <vector>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM{
  class GlmMvnPrior;
  class GammaModel;

  class GlmMvnSiginvIndSampler
    : public PosteriorSampler
  {
    typedef std::vector<Ptr<GammaModel> > Gvec;
  public:
    GlmMvnSiginvIndSampler(GlmMvnPrior *mod, Gvec ivar_pri);
    virtual void draw();
    virtual double logpri()const;
    uint dim()const;
  private:
    GlmMvnPrior *mod_;
    std::vector<Ptr<GammaModel> > ivar_pri_;
  };

}
#endif// BOOM_GLM_MVN_SIGINV_IND_SAMPLER_HPP
