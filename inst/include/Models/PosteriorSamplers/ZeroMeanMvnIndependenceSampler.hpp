/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_ZERO_MEAN_MVN_INDEPDENCE_SAMPLER_HPP_
#define BOOM_ZERO_MEAN_MVN_INDEPDENCE_SAMPLER_HPP_

#include <Models/ZeroMeanMvnModel.hpp>
#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM{

  // A prior that asserts the components of the ZeroMeanMvnModel are
  // independent, with conjugate Gamma marginal priors.
  //
  // This class models a single diagonal element in the variance
  // matrix of the ZeroMeanMvnModel.
  class ZeroMeanMvnIndependenceSampler
      : public PosteriorSampler {
   public:
    ZeroMeanMvnIndependenceSampler(ZeroMeanMvnModel *model,
                                   Ptr<GammaModelBase> prior,
                                   int which_variable);
    ZeroMeanMvnIndependenceSampler(ZeroMeanMvnModel *model,
                                   double prior_df,
                                   double sigma_guess,
                                   int which_variable);
    void draw();
    double logpri()const;
    void set_sigma_upper_limit(double max_sigma);
   private:
    ZeroMeanMvnModel *m_;
    Ptr<GammaModelBase> prior_;
    int which_variable_;
    double upper_truncation_point_;
  };

  // This class is a single sampler for all the diagonal elements of
  // Sigma.
  class ZeroMeanMvnCompositeIndependenceSampler
      : public PosteriorSampler {
   public:
    ZeroMeanMvnCompositeIndependenceSampler(
        ZeroMeanMvnModel *model,
        const std::vector<Ptr<GammaModelBase> > & siginv_priors,
        const Vec & sigma_upper_truncation_points);
    virtual void draw();
    virtual double logpri()const;
   private:
    ZeroMeanMvnModel *model_;
    std::vector<Ptr<GammaModelBase> > priors_;
    Vec sigma_upper_truncation_point_;
  };


}

#endif // BOOM_ZERO_MEAN_MVN_INDEPDENCE_SAMPLER_HPP_
