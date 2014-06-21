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

#ifndef BOOM_INDEPENDENT_MVN_CONJ_SAMPLER_HPP_
#define BOOM_INDEPENDENT_MVN_CONJ_SAMPLER_HPP_

#include <Models/IndependentMvnModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {

  // Posterior sampler for and MvnModel assuming the off-diagonal
  // elements of Sigma are all zero.  Draws each element of mu and
  // Sigma independently from the NormalInverseGamma distribution.
  class IndependentMvnConjSampler : public PosteriorSampler {
   public:
    IndependentMvnConjSampler(IndependentMvnModel *model,
                              const Vec &mean_guess,
                              const Vec & mean_sample_size,
                              const Vec &sd_guess,
                              const Vec &sd_sample_size,
                              const Vec &sigma_upper_limit);

    IndependentMvnConjSampler(IndependentMvnModel *model,
                              double mean_guess,
                              double mean_sample_size,
                              double sd_guess,
                              double sd_sample_size,
                              double sigma_upper_limit = infinity());

    virtual void draw();
    virtual double logpri()const;
   private:
    void check_sizes();
    void check_vector_size(const Vec &v, const char *vector_name);

    IndependentMvnModel *model_;
    Vec mean_prior_guess_;
    Vec mean_prior_sample_size_;
    Vec prior_ss_;
    Vec prior_df_;
    Vec sigma_upper_limit_;
  };

}
#endif// BOOM_INDEPENDENT_MVN_CONJ_SAMPLER_HPP_
