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
    GammaSuf *clone() const override;

    void set(double sum, double sumlog, double n);
    void clear() override;
    void Update(const DoubleData &dat) override;
    void update_raw(double y);
    void add_mixture_data(double y, double prob);

    // Add the given sufficient components to the sufficient statistics.
    void increment(double n, double sumy, double sumlog);

    double sum()const;
    double sumlog()const;
    double n()const;
    ostream & display(ostream &out)const;

    virtual void combine(Ptr<GammaSuf> s);
    virtual void combine(const GammaSuf & s);
    GammaSuf * abstract_combine(Sufstat *s) override;
    Vector vectorize(bool minimal=true)const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
					    bool minimal=true) override;
    Vector::const_iterator unvectorize(const Vector &v,
					    bool minimal=true) override;
    ostream &print(ostream &out)const override;
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
    GammaModelBase * clone()const override =0;

    virtual double alpha()const=0;
    virtual double beta()const=0;
    void add_mixture_data(Ptr<Data>, double prob) override;
    double pdf(Ptr<Data> dp, bool logscale) const override;
    double pdf(const Data * dp, bool logscale) const override;

    double Logp(double x, double &g, double &h, uint nd) const override ;
    double sim() const override;
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
    GammaModel *clone() const override;

    Ptr<UnivParams> Alpha_prm();
    Ptr<UnivParams> Beta_prm();
    const Ptr<UnivParams> Alpha_prm()const;
    const Ptr<UnivParams> Beta_prm()const;

    double alpha()const override;
    double beta()const override;
    void set_alpha(double);
    void set_beta(double);
    void set_params(double a, double b);

    double mean()const;

    // probability calculations
    double Loglike(const Vector &ab, Vector &g, Matrix &h, uint lev) const override;
    double loglikelihood(double a, double b) const;
    double loglikelihood_full(const Vector &ab, Vector *g, Matrix *h)const;
    void mle() override;
  };

}
#endif  // GAMMA_MODEL_H
