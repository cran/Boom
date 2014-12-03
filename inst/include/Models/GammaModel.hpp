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

#ifndef BOOM_GAMMA_MODEL_HPP
#define BOOM_GAMMA_MODEL_HPP

#include <iosfwd>
#include <cpputil/Ptr.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/DoubleModel.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>

//======================================================================
namespace BOOM{
  class GammaSuf: public SufstatDetails<DoubleData>{
  public:

    // constructor
    GammaSuf();
    GammaSuf(const GammaSuf &);
    GammaSuf *clone() const;

    void set(double sum, double sumlog, double n);
    void clear();
    void Update(const DoubleData &dat);
    void update_raw(double y);
    void add_mixture_data(double y, double prob);

    double sum()const;
    double sumlog()const;
    double n()const;
    ostream & display(ostream &out)const;

    virtual void combine(Ptr<GammaSuf> s);
    virtual void combine(const GammaSuf & s);
    GammaSuf * abstract_combine(Sufstat *s);
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    double sum_, sumlog_, n_;
  };
  //======================================================================
  class GammaModelBase // Gamma Model, Chi-Square Model, Scaled Chi-Square
    : public SufstatDataPolicy<DoubleData, GammaSuf>,
      public DiffDoubleModel,
      public NumOptModel,
      public EmMixtureComponent
  {
  public:
    GammaModelBase();
    GammaModelBase(const GammaModelBase &);
    virtual GammaModelBase * clone()const=0;

    virtual double alpha()const=0;
    virtual double beta()const=0;
    virtual void add_mixture_data(Ptr<Data>, double prob);
    double pdf(Ptr<Data> dp, bool logscale) const;
    double pdf(const Data * dp, bool logscale) const;

    double Logp(double x, double &g, double &h, uint nd) const ;
    double sim() const;
  };
  //======================================================================

  class GammaModel
    : public GammaModelBase,
      public ParamPolicy_2<UnivParams, UnivParams>,
      public PriorPolicy
  {
  public:
    // The usual parameterization of the Gamma distribution a =
    // shape, b = scale, mean = a/b.
    GammaModel(double a=1.0, double b=1.0);

    // To initialize a GammaModel with shape (a) and mean parameters,
    // simply include a third argument that is an int.
    GammaModel(double shape, double mean, int);

    GammaModel(const GammaModel &m);
    GammaModel *clone() const;

    Ptr<UnivParams> Alpha_prm();
    Ptr<UnivParams> Beta_prm();
    const Ptr<UnivParams> Alpha_prm()const;
    const Ptr<UnivParams> Beta_prm()const;

    double alpha()const;
    double beta()const;
    void set_alpha(double);
    void set_beta(double);
    void set_params(double a, double b);

    double mean()const;

    // probability calculations
    double Loglike(const Vector &ab, Vec &g, Mat &h, uint lev) const;
    double loglikelihood(double a, double b) const;
    double loglikelihood_full(const Vec &ab, Vec *g, Mat *h)const;
    void mle();
  };

}
#endif  // GAMMA_MODEL_H
