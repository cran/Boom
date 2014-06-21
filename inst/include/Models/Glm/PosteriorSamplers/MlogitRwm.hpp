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

#ifndef BOOM_MLOGIT_RWM_SAMPLER_HPP
#define BOOM_MLOGIT_RWM_SAMPLER_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/MvnBase.hpp>

namespace BOOM{
  class MlogitRwm : public PosteriorSampler{
  public:
    MlogitRwm(MultinomialLogitModel *mlm, Ptr<MvnBase> pri);
    MlogitRwm(MultinomialLogitModel *mlm,
	      const Vec &mu,
	      const Spd & Ominv);
    virtual void draw();
    virtual double logpri()const;
  private:
    MultinomialLogitModel *mlm_;
    Ptr<MvnBase> pri_;
    Vec mu,g, b, bstar;
    Spd H, ivar;
  };

}

#endif// BOOM_MLOGIT_RWM_SAMPLER_HPP
