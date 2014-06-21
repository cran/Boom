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
#ifndef BOOM_CORRELATION_SAMPLER_HPP
#define BOOM_CORRELATION_SAMPLER_HPP
#include <Samplers/Sampler.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <TargetFun/TargetFun.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/MvnModel.hpp>
namespace BOOM{

  class CorrTF
    : public TargetFun
  {
  public:
    CorrTF(Ptr<SpdParams>,
	   Ptr<LoglikeModel>,
	   Ptr<CorrModel>);
    CorrTF(const CorrTF &rhs);
    //    CorrTF * clone()const;
    virtual double operator()(const Vec &x)const;
  private:
    mutable Corr R, R2;
    mutable Ptr<SpdParams> spd;
    mutable Ptr<LoglikeModel> mod;
    mutable Vec wsp;
    Ptr<CorrModel> pri;
    bool refresh;
  };

  // Draws from the posterior distribution of the correlation matrix
  // in a Gaussian model with known means and variances given a
  // CorrModel prior distribution on the correlation matrix
  class MvnCorrelationSampler
    : public PosteriorSampler
  {
  public:
    MvnCorrelationSampler(MvnModel *,
			  Ptr<CorrModel>,
			  bool refresh_suf = false);
    void draw();
    double logpri()const;
  private:
    double logp(double r);
    void find_limits();
    void draw_one();
    double Rdet(double r);
    void set_r(double r);
    void check_limits(double oldr, double eps);

    MvnModel *mod_;       // supplies likelihood
    Ptr<CorrModel> pri_;      // prior for R
    Corr R_;                  // workspace
    Spd Sumsq_;
    double df_;
    int i_, j_;
    double lo_, hi_;
  };

}
#endif// BOOM_CORRELATION_SAMPLER_HPP
