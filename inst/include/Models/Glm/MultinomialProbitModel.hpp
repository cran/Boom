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

#ifndef BOOM_MULTINOMIAL_PROBIT_MODEL_HPP
#define BOOM_MULTINOMIAL_PROBIT_MODEL_HPP

#include <Models/Glm/Glm.hpp>   // for GlmCoefs
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Glm/MvReg2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Glm/ChoiceData.hpp>

namespace BOOM{
  class TrunMvnTF;
  class MultinomialProbitModel
    : public ParamPolicy_2<GlmCoefs, SpdParams>,
      public IID_DataPolicy<ChoiceData>,
      public PriorPolicy,
      public LatentVariableModel
  {
  public:
    typedef std::vector<Ptr<CategoricalData> > ResponseVec;
    enum ImputationMethod{Slice, Gibbs};

    // each column of beta_subject corresponds to a different choice.
    MultinomialProbitModel(const Mat & beta_subject,
			   const Vec & beta_choice,
			   const Spd & utility_covariance);

//     // the function make_catdat_ptrs can make a ResponseVec out of a
//     // vector of strings or uints
//     MultinomialProbitModel(ResponseVec responses,
// 			  const Mat &Xsubject_info,
// 			  const Arr3 &Xchoice_info);
//     // dim(Xchoice_info) = [#obs, #choices, #choice x's]

//     MultinomialProbitModel(ResponseVec responses,    // no choice information
// 			  const Mat &Xsubject_info);

    MultinomialProbitModel(const std::vector<Ptr<ChoiceData> > &);
    MultinomialProbitModel(const MultinomialProbitModel &rhs);
    MultinomialProbitModel * clone()const;

    void use_slice_sampling(){imp_method = Slice;}
    void use_Gibbs_sampling(){imp_method = Gibbs;}
    virtual void impute_latent_data(RNG &rng);
    virtual double complete_data_loglike()const;

    double pdf(Ptr<Data> dp, bool logscale)const;
    double pdf(Ptr<ChoiceData> dp, bool logscale)const;
    virtual void initialize_params();

    const Vec & beta()const;
    Vec beta_subject(uint choice)const;
    Vec beta_choice()const;

    const Spd & Sigma()const;
    const Spd & siginv()const;
    double ldsi()const;

    // eta is the value of the linear predictor when evaluated at X
    Vec eta(Ptr<ChoiceData>)const;
    Vec &eta(Ptr<ChoiceData>, Vec &ans)const;

    uint n()const;
    uint xdim()const;
    uint subject_nvars()const;
    uint choice_nvars()const;
    uint Nchoices()const;

    void set_beta(const Vec &b);
    void set_included_coefficients(const Vec &b);
    void set_Sigma(const Spd &Sig);
    void set_siginv(const Spd &siginv);

    Ptr<GlmCoefs> Beta_prm(){return ParamPolicy::prm1();}
    const Ptr<GlmCoefs> Beta_prm()const{return ParamPolicy::prm1();}
    Ptr<SpdParams> Sigma_prm(){return ParamPolicy::prm2();}
    const Ptr<SpdParams> Sigma_prm()const{return ParamPolicy::prm2();}

    const Spd &xtx()const;
    const Spd &yyt()const;
    double yty()const;
    const Vec &xty()const;

    virtual void add_data(Ptr<Data>);
    virtual void add_data(Ptr<ChoiceData>);
  private:
    ImputationMethod imp_method;
    mutable Vec wsp;
    std::vector<Vec> U;
    uint nchoices_, subject_xdim_, choice_xdim_;
    Spd yyt_;   // sum y*y^T
    Spd xtx_;   // sum
    Vec xty_;

    Ptr<GlmCoefs> make_beta(const Mat &beta_subject, const Vec & beta_choice);
    Ptr<GlmCoefs> make_beta(const std::vector<Ptr<ChoiceData> > &);
    void setup_suf();
    void impute_u(RNG &rng, Vec &u, Ptr<ChoiceData>, TrunMvnTF & );
    void impute_u_slice(Vec &u, Ptr<ChoiceData>, TrunMvnTF & );
    void impute_u_Gibbs(RNG &rng, Vec &u, Ptr<ChoiceData>, TrunMvnTF & );
    void update_suf(const Vec & u, Ptr<ChoiceData>);
  };


}
#endif// BOOM_MULTINOMIAL_PROBIT_MODEL_HPP
