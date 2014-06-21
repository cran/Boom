/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef CUMULATIVE_LOGIT_MODEL_HPP
#define CUMULATIVE_LOGIT_MODEL_HPP

#include <Models/Glm/Glm.hpp>
#include <Models/Glm/OrdinalCutpointModel.hpp>

namespace BOOM{
  class CumulativeLogitModel
      : public OrdinalCutpointModel
  {
   public:
    CumulativeLogitModel(const Vec &beta, const Vec & delta);
    CumulativeLogitModel(const Mat &X, const Vec &y);
    CumulativeLogitModel(const CumulativeLogitModel &rhs);
    virtual CumulativeLogitModel * clone()const;

    virtual double link_inv(double)const;
    virtual double dlink_inv(double)const;
   private:
    virtual double simulate_latent_variable()const;
  };

} // ends namespace BOOM

#endif // CUMULATIVE_LOGIT_MODEL_HPP
