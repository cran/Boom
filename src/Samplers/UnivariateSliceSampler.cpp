/*
  Copyright (C) 2006 Steven L. Scott

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
#include <Samplers/UnivariateSliceSampler.hpp>
#include <distributions.hpp>
#include <cmath>
#include <cassert>
#include <cpputil/math_utils.hpp>

namespace BOOM{
  typedef UnivariateSliceSampler USS;

USS::UnivariateSliceSampler(const Target &logpost,
                            int dim,
                            double suggested_dx,
                            bool unimodal,
                            RNG *rng)
    : Sampler(rng),
      f_(logpost),
      theta_(dim)
  {
    for (int i = 0; i < dim; ++i) {
      scalar_targets_.push_back(ScalarTargetFunAdapter(f_, &theta_, i));
      scalar_samplers_.push_back(ScalarSliceSampler(
          scalar_targets_.back(),
          unimodal,
          suggested_dx,
          rng));
    }
  }

  Vector USS::draw(const Vector &x){
    theta_ = x;
    for (int i = 0; i < scalar_samplers_.size(); ++i) {
      theta_[i] = scalar_samplers_[i].draw(theta_[i]);
    }
    return theta_;
  }

}  // namespace BOOM
