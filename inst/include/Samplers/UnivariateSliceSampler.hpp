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

#include <Samplers/Sampler.hpp>
#include <TargetFun/TargetFun.hpp>
#include <LinAlg/Vector.hpp>

namespace BOOM{

  /*
   * A "Univariate" slice sampler draws a vector one component at a
   * time.  It is not a "scalar" slice sampler.
   */

  class UnivariateSliceSampler : public Sampler{
  public:
    UnivariateSliceSampler(const TargetFun &F, bool unimodal=false);
    Vec draw(const Vec &x);
    double logp(const Vec &x)const;
  private:
    const TargetFun &f;
    bool unimodal;
    Vec theta, wsp;
    uint which;
    double lo, hi; // slice boundaries
    double plo, phi; // logp at slice boundaries
    double y, pstar; // current value and slice height

    void doubling(bool);
    void contract(double lam, double p);
    void find_limits();
    void initialize(); // draws vertical slice. ensures hi,lo are valid
    void draw_1();
    double f1(double x);
    void validate(double &value, double &prob, double anchor);
  };

}
