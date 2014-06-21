/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MARKOV_CONJUGATE_SAMPLER_HPP
#define BOOM_MARKOV_CONJUGATE_SAMPLER_HPP

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/MarkovModel.hpp>
#include <Models/ProductDirichletModel.hpp>
#include <Models/DirichletModel.hpp>
#include <distributions/rng.hpp>

namespace BOOM{
  class MarkovConjSampler
    : public PosteriorSampler
  {
  public:
    MarkovConjSampler(MarkovModel * Mod,
		      Ptr<ProductDirichletModel> Q,
		      Ptr<DirichletModel> pi0);
    MarkovConjSampler(MarkovModel * Mod,
		      Ptr<ProductDirichletModel> Q);

    MarkovConjSampler(MarkovModel * Mod,
                      const Mat & Nu);
    MarkovConjSampler(MarkovModel * Mod,
                      const Mat & Nu,
                      const Vec & nu);

    virtual MarkovConjSampler * clone()const;

    double logpri()const;
    void draw();
    void find_posterior_mode();

    const Mat & Nu()const;
    const Vec & nu()const;  // throws if pi0_ is not set
  private:
    MarkovModel * mod_;
    Ptr<ProductDirichletModel> Q_;
    Ptr<DirichletModel> pi0_;
   protected:
    void check_pi0()const;
    void check_nu()const;
    mutable Vec wsp;
  };
}
#endif// BOOM_MARKOV_CONJUGATE_SAMPLER_HPP
