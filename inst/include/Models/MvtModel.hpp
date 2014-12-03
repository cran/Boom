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

#ifndef MVT_MODEL_H
#define MVT_MODEL_H
#include <LinAlg/Types.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/VectorModel.hpp>
#include <Models/SpdParams.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <distributions/rng.hpp>

namespace BOOM{
  class ScaledChisqModel;
  class WeightedMvnModel;

  class MvtModel
    : public CompositeParamPolicy,
      //ParamPolicy_3<VectorParams, SpdParams, UnivParams>,
      public IID_DataPolicy<VectorData>,
      public PriorPolicy,
      public LatentVariableModel,
      public LoglikeModel,
      public LocationScaleVectorModel
  {
  public:
    MvtModel(uint p, double mu=0.0, double sig=1.0, double nu =30.0);
    MvtModel(const Vec &mean, const Spd &Var, double Nu);
    MvtModel(const MvtModel &m);

    MvtModel *clone() const;

    void initialize_params();

    Ptr<VectorParams> Mu_prm();
    Ptr<SpdParams> Sigma_prm();
    Ptr<UnivParams> Nu_prm();

    const Ptr<VectorParams> Mu_prm()const;
    const Ptr<SpdParams> Sigma_prm()const;
    const Ptr<UnivParams> Nu_prm()const;

    int dim()const;
    const Vec &mu()const;
    const Spd &Sigma()const;
    const Spd &siginv()const;
    double ldsi()const;
    double nu() const;

    void set_mu(const Vec &);
    void set_Sigma(const Spd &);
    void set_siginv(const Spd &);
    void set_S_Rchol(const Vec &S, const Mat &L);
    void set_nu(double);

    double logp(const Vec &x)const;

    double pdf(Ptr<VectorData>, bool logscale)const;
    double pdf(dPtr dp, bool logscale) const;
    double pdf(const Vec &x, bool logscale) const;

    virtual void add_data(Ptr<Data>);
    virtual void add_data(Ptr<VectorData>);

    void mle();  // ECME
    virtual double loglike(const Vector &mu_siginv_triangle_nu)const;
    virtual void impute_latent_data(RNG &rng);
    void Estep();  // E step for EM/ECME

    virtual double complete_data_loglike()const;
    Vec sim()const;
  private:
    void Impute(bool sample, RNG &rng = GlobalRng::rng);
    Ptr<WeightedMvnModel> mvn;
    Ptr<ScaledChisqModel> wgt;
  };


}
#endif // MVT_MODEL_H
