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

#ifndef BOOM_VS_SEP_STRAT_SAMPLER
#define BOOM_VS_SEP_STRAT_SAMPLER

#include <Models/Glm/GlmMvnPrior.hpp>
#include <Models/GammaModel.hpp>
#include <vector>

namespace BOOM{
  class GlmMvnSiginvSepStratSampler
    : public PosteriorSampler{
  public:
    // S_pri specifies a set of independent gamma priors on the
    // marginal inverse variances.  A marginally uniform prior is
    // assumed for the correlations.

    GlmMvnSiginvSepStratSampler(GlmMvnPrior *mod,
				std::vector<Ptr<GammaModel> > S_pri);
    virtual void draw();
    virtual double logpri()const;
    uint dim()const;

  private:
    GlmMvnPrior *mod_;
    std::vector<Ptr<GammaModel> > Spri;
    Mat LT;
    Mat Wsp;
    Mat sumsq_chol;
    Vec nobs;
    Spd Sigma;
    Vec S;

    void draw_L(uint i, uint j);
    void set_Sigma();
    void observe_sigma(const Spd &);

    mutable bool L_current;

    //-------------------------------------------------
  };
}
#endif // BOOM_VS_SEP_STRAT_SAMPLER
