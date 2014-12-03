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

#ifndef BOOM_T_REGRESSION_HPP
#define BOOM_T_REGRESSION_HPP

#include <Models/Glm/Glm.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  class ScaledChisqModel;
  class WeightedRegressionModel;
  class TRegressionModel
    : public GlmModel,
      public CompositeParamPolicy,
      public IID_DataPolicy<RegressionData>,
      public PriorPolicy,
      public NumOptModel,
      public LatentVariableModel
  {
  public:

    TRegressionModel(uint p);   // dimension of beta
    TRegressionModel(const Vec &b, double Sigma, double nu=30);
    TRegressionModel(const TRegressionModel &rhs);
    TRegressionModel(const DatasetType &d, bool inc_all=true);
    TRegressionModel(const Mat &X, const Vec &y);
    TRegressionModel * clone()const;

    virtual GlmCoefs & coef();
    virtual const GlmCoefs & coef()const;
    virtual Ptr<GlmCoefs> coef_prm();
    virtual const Ptr<GlmCoefs> coef_prm()const;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm()const;
    Ptr<UnivParams> Nu_prm();
    const Ptr<UnivParams> Nu_prm()const;

    //    void set_beta(const Vec &b){wreg_->set_beta(b);}
    void set_sigsq(double s2);
    void set_nu(double Nu);

    // beta() and Beta() inherited from GLM;
    const double & sigsq()const;
    double sigma()const;
    const double & nu()const;

    // The argument to Loglike is a vector containing the included
    // regression coefficients, followed by the residual 'dispersion'
    // parameter sigsq, followed by the tail thickness parameter nu.
    double Loglike(const Vector &beta_sigsq_nu,
                   Vec &g, Mat &h, uint nd)const;
    void mle();
    double complete_data_Loglike(Vec &g, Mat &h, uint nd)const;
    virtual double complete_data_loglike()const;

    void impute_latent_data(RNG &rng);
    void initialize_params();
    void EStep();

    double pdf(dPtr, bool)const;
    double pdf(Ptr<DataType>, bool)const;

    Ptr<RegressionData>  simdat()const;
    Ptr<RegressionData>  simdat(const Vec &X)const;

    virtual void add_data(Ptr<RegressionData>);
    virtual void add_data(Ptr<Data>);

  private:
    void Impute(bool draw, RNG &rng); //
    Ptr<WeightedRegressionModel> wreg_;
    Ptr<ScaledChisqModel> wgt_;
    void setup_params();
  };
  //------------------------------------------------------------



}

#endif// BOOM_T_REGRESSION_HPP
