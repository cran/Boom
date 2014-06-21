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
#ifndef BOOM_LOGIT_HOLMES_HELD_SAMPLER_HPP
#define BOOM_LOGIT_HOLMES_HELD_SAMPLER_HPP

#include <Models/Glm/LogisticRegressionModel.hpp>
#include <Models/MvnBase.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/Glm/WeightedRegressionModel.hpp>
namespace BOOM{

  class WeightedRegSuf;


// posterior sampler to draw from the posterior distribution of a
// multinomical logit model using Holmes and Held's data augmentation
// strategy
  class LogitSampler
    : public PosteriorSampler
  {
  public:
    LogitSampler(LogisticRegressionModel *mod,
		 Ptr<MvnBase> pri);
    virtual void draw();
    virtual double logpri()const;
    void impute_latent_data();
    const Ptr<WeightedRegSuf> suf()const{return suf_;}

  private:
    double draw_z(bool y, double eta)const;
    double draw_lambda(double r)const;
    void draw_beta();

    LogisticRegressionModel *mod_;
    Ptr<MvnBase> pri_;
    Ptr<WeightedRegSuf> suf_;

    Spd ivar;     // workspace:  stores inverse variance
    Vec ivar_mu;  // workspace:  stores un-normalized mean
  };

}

#endif // BOOM_LOGIT_HOLMES_HELD_SAMPLER_HPP
