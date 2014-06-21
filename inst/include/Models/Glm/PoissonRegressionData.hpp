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
#ifndef BOOM_POISSON_REGRESSION_DATA_HPP_
#define BOOM_POISSON_REGRESSION_DATA_HPP_

#include <Models/Glm/Glm.hpp>

namespace BOOM {
  class PoissonRegressionData : public GlmData<IntData> {
   public:
    PoissonRegressionData(int y, const Vec &x);
    PoissonRegressionData(int y, const Vec &x, double exposure);
    virtual PoissonRegressionData * clone()const;
    virtual ostream & display(ostream &out)const;
    double exposure()const;
    double log_exposure()const;
   private:
    double exposure_;
    double log_exposure_;
    // saving both exposure and log_exposure keeps us from computing
    // the log of the same thing over and over again in the sampler.
  };
}  // namespace BOOM

#endif  //  BOOM_POISSON_REGRESSION_DATA_HPP_
