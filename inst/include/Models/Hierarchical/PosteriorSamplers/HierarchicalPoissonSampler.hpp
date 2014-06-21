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

#ifndef BOOM_HIERARCHICAL_POISSON_POSTERIOR_SAMPLER_HPP_
#define BOOM_HIERARCHICAL_POISSON_POSTERIOR_SAMPLER_HPP_

#include <Models/DoubleModel.hpp>
#include <Models/Hierarchical/HierarchicalPoissonModel.hpp>

namespace BOOM {

  class HierarchicalPoissonSampler : public PosteriorSampler {
   public:
    HierarchicalPoissonSampler(HierarchicalPoissonModel *model,
                               Ptr<DoubleModel> gamma_mean_prior,
                               Ptr<DoubleModel> gamma_sample_size_prior);
    virtual double logpri()const;
    virtual void draw();
   private:
    HierarchicalPoissonModel *model_;
    Ptr<DoubleModel> gamma_mean_prior_;
    Ptr<DoubleModel> gamma_sample_size_prior_;
  };

}  // namespace BOOM

#endif //  BOOM_HIERARCHICAL_POISSON_POSTERIOR_SAMPLER_HPP_
