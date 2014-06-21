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
#ifndef BOOM_MULTINOMIAL_DIRICHLET_SAMPLER_HPP
#define BOOM_MULTINOMIAL_DIRICHLET_SAMPLER_HPP
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
namespace BOOM{

  class MultinomialModel;
  class DirichletModel;

  class MultinomialDirichletSampler
    : public PosteriorSampler
  {
  public:
    MultinomialDirichletSampler(MultinomialModel *mod, const Vec & nu);

    MultinomialDirichletSampler(MultinomialModel *mod, Ptr<DirichletModel>);

    MultinomialDirichletSampler(const MultinomialDirichletSampler &rhs);
    MultinomialDirichletSampler * clone()const;

    void draw();
    double logpri()const;
    void find_posterior_mode();
  private:
    MultinomialModel *mod_;
    Ptr<DirichletModel> pri_;
  };

}
#endif// BOOM_MULTINOMIAL_DIRICHLET_SAMPLER_HPP
