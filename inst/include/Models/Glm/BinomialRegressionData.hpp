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
#ifndef BOOM_BINOMIAL_REGRESSION_DATA_HPP_
#define BOOM_BINOMIAL_REGRESSION_DATA_HPP_

#include <Models/Glm/Glm.hpp>

namespace BOOM{
  class BinomialRegressionData
      : public GlmData<IntData>{
   public:
    typedef GlmData<IntData> Base;
    BinomialRegressionData(uint y, uint n, const Vec &x, bool add_icpt=false);
    virtual BinomialRegressionData * clone()const;
    void set_n(uint n, bool check = true);
    void set_y(uint y, bool check = true);
    uint n()const;
    void check()const;  // throws if n < y
    virtual ostream & display(ostream &out)const;
   private:
    uint n_;  // number of trials in the binomial process
              // y() is the number of successes
  };
}
#endif // BOOM_BINOMIAL_REGRESSION_DATA_HPP_
