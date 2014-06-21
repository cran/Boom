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

#ifndef BOOM_DIRICHLET_POSTERIOR_SAMPLER_HPP
#define BOOM_DIRICHLET_POSTERIOR_SAMPLER_HPP

#include <Models/DoubleModel.hpp>
#include <Models/VectorModel.hpp>
#include <Models/DirichletModel.hpp>
#include <TargetFun/TargetFun.hpp>

namespace BOOM{
  class DirichletPosteriorSampler
    : public PosteriorSampler{
  public:
    DirichletPosteriorSampler(DirichletModel *m,
                              Ptr<VectorModel> phi,
                              Ptr<DoubleModel> alpha);

    virtual void draw();
    virtual double logpri()const;
    uint dim()const;
  private:
    DirichletModel *mod_;  // param is nu = alpha * phi, where alpha = sum(nu)
    Ptr<VectorModel> phi_prior_;
    Ptr<DoubleModel> alpha_prior_;
  };

//----------------------------------------------------------------------
  class DirichletLogp
    : public ScalarTargetFun
  {
  public:
    DirichletLogp(uint pos,  const Vec & nu, const Vec & sumlogpi, double nobs,
		  Ptr<VectorModel> phi, Ptr<DoubleModel> alpha, double min_nu=0);
    double operator()(double nu)const;

  private:
    double logp()const;
    const Vec & sumlogpi_;
    const double nobs_;
    const uint pos_;
    mutable Vec nu_;
    const double min_nu_;

    Ptr<DoubleModel> alpha_prior_;
    Ptr<VectorModel> phi_prior_;
  };

}
#endif// BOOM_DIRICHLET_POSTERIOR_SAMPLER_HPP
