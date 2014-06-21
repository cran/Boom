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
#ifndef BOOM_MARKOV_DIST_HPP
#define BOOM_MARKOV_DIST_HPP

#include <LinAlg/Types.hpp>
#include <BOOM.hpp>
#include <vector>

namespace BOOM{
  class Selector;
  using BOOM::uint;

  // returns the stationary distribution of the transition matrix Q.
  // Each row of Q sums to 1.
  Vec get_stat_dist(const Mat &Q);

  // returns the probability that state r happens before state s in
  // a Markov chain with initial distribution pi0 and transition
  // matrix P.
  double preceeds(uint r, uint s,  const Vec &pi0, const Mat &P);

  // returns the probability that any of the states in r happen before
  // any of the states in s in a Markov chain with initial
  // distribution pi0 and transition matrix P
  double preceeds(const Selector &r, const Selector &s,
                  const Vec &pi0, const Mat &P);

  // On input P is the SxS matrix of absorption probabilities abs
  // indicates the absorbing states.  The output is a matrix with
  // S-|abs| rows and |abs| columns.  Each row is a probability
  // distribution giving the conditional probability of being absorbed
  // into a particular state.
  Mat compute_conditional_absorption_probs(const Mat &P, const Selector &abs);
}
#endif// BOOM_MARKOV_DIST_HPP
