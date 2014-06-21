/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_ZERO_MEAN_MVN_MODEL_HPP_
#define BOOM_ZERO_MEAN_MVN_MODEL_HPP_
#include <Models/MvnBase.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/PosteriorSamplers/ZeroMeanMvnConjSampler.hpp>

namespace BOOM{

  class ZeroMeanMvnConjSampler;

  class ZeroMeanMvnModel
      : public MvnBase,
        public LoglikeModel,
        public ParamPolicy_1<SpdParams>,
        public SufstatDataPolicy<VectorData, MvnSuf>,
        public ConjugatePriorPolicy<ZeroMeanMvnConjSampler>
  {
   public:
    ZeroMeanMvnModel(int dim);
    virtual ZeroMeanMvnModel * clone()const;
    virtual const Vec & mu() const;
    virtual const Spd & Sigma()const;
    virtual void set_Sigma(const Spd &);
    virtual const Spd & siginv() const;
    virtual void set_siginv(const Spd &);
    virtual double ldsi()const;

    virtual void mle();
    virtual double loglike()const;
    virtual double pdf(Ptr<Data>, bool logscale)const;

    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm()const;
   private:
    Vec mu_;
  };
}
#endif// BOOM_ZERO_MEAN_MVN_MODEL_HPP_
