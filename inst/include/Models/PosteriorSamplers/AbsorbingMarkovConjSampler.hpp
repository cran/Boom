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

#ifndef BOOM_ABSORBING_MARKOV_CONJUGATE_SAMPLER_HPP
#define BOOM_ABSORBING_MARKOV_CONJUGATE_SAMPLER_HPP

#include <Models/MarkovModel.hpp>
#include <Models/PosteriorSamplers/MarkovConjSampler.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{

  class AbsorbingMarkovConjSampler
    : public MarkovConjSampler
  {
  public:
    AbsorbingMarkovConjSampler(MarkovModel * Mod,
                               Ptr<ProductDirichletModel> Q,
                               Ptr<DirichletModel> pi0,
                               std::vector<uint> absorbing_states);
    AbsorbingMarkovConjSampler(MarkovModel * Mod,
                               Ptr<ProductDirichletModel> Q,
                               std::vector<uint> absorbing_states);
    AbsorbingMarkovConjSampler(MarkovModel * Mod,
                               const Mat & Nu,
                               std::vector<uint> absorbing_states);
    AbsorbingMarkovConjSampler(MarkovModel * Mod,
                               const Mat & Nu,
                               const Vec & nu,
                               std::vector<uint> absorbing_states);

    virtual AbsorbingMarkovConjSampler * clone()const;

    virtual double logpri()const;
    virtual  void draw();
    virtual void find_posterior_mode();

  private:
    MarkovModel * mod_;
    Selector abs_;
    Selector trans_;
  };


}

#endif// BOOM_ABSORBING_MARKOV_CONJUGATE_SAMPLER_HPP
