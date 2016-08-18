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

#ifndef MVN_MODEL_H
#define MVN_MODEL_H


#include <Models/ModelTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/MvnBase.hpp>
#include <Models/MvnGivenSigma.hpp>
#include <Models/PosteriorSamplers/MvnConjSampler.hpp>
#include <Models/WishartModel.hpp>

namespace BOOM{

  class MvnModel:
    public MvnBaseWithParams,
    public LoglikeModel,
    public SufstatDataPolicy<VectorData, MvnSuf>,
    public PriorPolicy,
    public EmMixtureComponent
  {
  public:
    typedef MvnBaseWithParams Base;
    // A p-dimensional MvnModel, with a constant value mu for a mean,
    // and a diagonal variance matrix sig^2.  Note that the
    // constructor expects the standard deviation instead of a
    // variance.
    MvnModel(uint p, double mu=0.0, double sig=1.0);   // N(mu.1, diag(sig^2))

    // Use this constructor if you want to directly specify the mean
    // and variance.  If the third argument 'ivar' is 'true' then you
    // specify the mean and the precision ('ivar' is inverse
    // variance).
    MvnModel(const Vector &mean, const SpdMatrix &V,
             bool ivar=false);

    // Use this constructor if you already have pointers to model
    // parameters.  This is useful if the model is supposed to share
    // parameters with another model, e.g. a mixture of normals with a
    // common variance parameter.
    MvnModel(Ptr<VectorParams> mu, Ptr<SpdParams> Sigma);

    // Use this constructor if you have a set of multivariate normal
    // observations.  It sets the initial parameter values to the MLE.
    MvnModel(const std::vector<Vector> &data);

    MvnModel(const MvnModel &m);
    MvnModel *clone() const override;

    void mle() override;
    void initialize_params() override;
    void add_mixture_data(Ptr<Data>, double prob) override;
    double loglike(const Vector &mu_siginv) const override;

    void add_raw_data(const Vector &y);
    double pdf(Ptr<Data>, bool logscale)const;
    double pdf(const Data *, bool logscale)const override;
    double pdf(const Vector &x, bool logscale)const;

    void set_conjugate_prior(Ptr<MvnGivenSigma>, Ptr<WishartModel>);
    void set_conjugate_prior(Ptr<MvnConjSampler>);

    Vector sim()const override;
    Vector sim(RNG &rng)const;
  };
  //------------------------------------------------------------
}
#endif  // MVN_MODEL_H
