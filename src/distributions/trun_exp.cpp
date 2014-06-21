/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#include <distributions.hpp>

namespace BOOM{

double rtrun_exp_mt(RNG & rng, double lam, double lo, double hi){

  // samples a random variable from the exponential distribution with
  // rate lam, with support truncated between lo and hi


  double Fmax = 1-exp(-lam*(hi-lo));
  double u = runif_mt(rng, 0, 1);

  double x = lo - log(1-u*Fmax)/lam;
  return x;
}

double rtrun_exp(double lam, double lo, double hi){
  return rtrun_exp_mt(GlobalRng::rng, lam, lo, hi); }

}
