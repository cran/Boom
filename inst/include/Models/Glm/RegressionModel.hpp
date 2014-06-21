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

#ifndef REGRESSIION_MODEL_H
#define REGRESSIION_MODEL_H

#include <BOOM.hpp>
#include <Models/Glm/Glm.hpp>
#include <LinAlg/QR.hpp>
#include <Models/Sufstat.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>

namespace BOOM{

  class RegressionConjSampler;
  class DesignMatrix;
  class MvnGivenXandSigma;
  class GammaModel;

  class AnovaTable{
   public:
    double SSE, SSM, SST;
    double MSM, MSE;
    double df_error, df_model, df_total;
    double F, p_value;
    ostream & display(ostream &out)const;
  };

  ostream & operator<<(ostream &out, const AnovaTable &tab);

  Mat add_intercept(const Mat &X);
  Vector add_intercept(const Vector &X);

  //------- virtual base for regression sufficient statistics ----
  class RegSuf: virtual public Sufstat{
  public:
    typedef std::vector<Ptr<RegressionData> > dataset_type;
    typedef Ptr<dataset_type, false> dsetPtr;

    RegSuf * clone()const=0;

    virtual uint size()const=0;  // dimension of beta
    virtual double yty()const=0;
    virtual Vector xty()const=0;
    virtual Spd xtx()const=0;

    virtual Vector xty(const Selector &)const=0;
    virtual Spd xtx(const Selector &)const=0;

    // return least squares estimates of regression params
    virtual Vector beta_hat()const=0;
    virtual double SSE()const=0;  // SSE measured from ols beta
    virtual double SST()const=0;
    virtual double ybar()const=0;
    virtual double n()const=0;

    AnovaTable anova()const;

    virtual void add_mixture_data(double y, const Vector &x, double prob)=0;
    virtual void add_mixture_data(double y, const ConstVectorView &x, double prob)=0;
    virtual void combine(Ptr<RegSuf>)=0;

    virtual ostream &print(ostream &out)const;
  };
  inline ostream & operator<<(ostream &out, const RegSuf &suf){
    return suf.print(out);
  }
  //------------------------------------------------------------------
  class QrRegSuf :
    public RegSuf,
    public SufstatDetails<RegressionData>
  {
    mutable QR qr;
    mutable Vector Qty;
    mutable double sumsqy;
    mutable bool current;
  public:
    QrRegSuf(const Mat &X, const Vector &y);
    QrRegSuf(const QrRegSuf &rhs);  // value semantics

    QrRegSuf *clone()const;
    virtual void clear();
    virtual void Update(const DataType &);
    virtual void add_mixture_data(double y, const Vector &x, double prob);
    virtual void add_mixture_data(double y, const ConstVectorView &x, double prob);
    virtual uint size()const;  // dimension of beta
    virtual double yty()const;
    virtual Vector xty()const;
    virtual Spd xtx()const;

    virtual Vector xty(const Selector &)const;
    virtual Spd xtx(const Selector &)const;

    virtual Vector beta_hat()const;
    virtual Vector beta_hat(const Vector &y)const;
    virtual double SSE()const;
    virtual double SST()const;
    virtual double ybar()const;
    virtual double n()const;
    void refresh_qr(const std::vector<Ptr<DataType> > &) const ;
    //    void check_raw_data(const Mat &X, const Vector &y);
    virtual void combine(Ptr<RegSuf>);
    virtual void combine(const RegSuf &);
    QrRegSuf * abstract_combine(Sufstat *s);

    virtual Vector vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal = true);
    virtual Vec::const_iterator unvectorize(const Vector &v,
					    bool minimal = true);
    virtual ostream &print(ostream &out)const;
  };
  //------------------------------------------------------------------
  class NeRegSuf
    : public RegSuf,
      public SufstatDetails<RegressionData>
  {   // directly solves 'normal equations'
  public:
    // An empty, but right-sized set of sufficient statistics.
    NeRegSuf(uint p);

    // Build from the design matrix X and response vector y.
    NeRegSuf(const Mat &X, const Vector &y);

    // Build from the indiviudal sufficient statistic components.  The
    // 'n' is needed because X might not have an intercept term.
    NeRegSuf(const Spd &xtx, const Vector &xty, double yty, double n);

    // Build from a sequence of Ptr<RegressionData>
    template <class Fwd> NeRegSuf(Fwd b, Fwd e);
    NeRegSuf(const NeRegSuf &rhs);
    NeRegSuf *clone()const;

    // If fixed, then xtx will not be changed by a call to clear(),
    // add_mixture_data(), or any of the flavors of Update().
    void fix_xtx(bool tf = true);

    virtual void clear();
    virtual void add_mixture_data(
        double y, const Vector &x, double prob);
    virtual void add_mixture_data(
        double y, const ConstVectorView &x, double prob);
    virtual void Update(const RegressionData & rdp);
    virtual uint size()const;  // dimension of beta
    virtual double yty()const;
    virtual Vector xty()const;
    virtual Spd xtx()const;
    virtual Vector xty(const Selector &)const;
    virtual Spd xtx(const Selector &)const;
    virtual Vector beta_hat()const;
    virtual double SSE()const;
    virtual double SST()const;
    virtual double ybar()const;
    virtual double n()const;
    virtual void combine(Ptr<RegSuf>);
    virtual void combine(const RegSuf &);
    NeRegSuf * abstract_combine(Sufstat *s);

    virtual Vector vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(
        Vec::const_iterator &v, bool minimal=true);
    virtual Vec::const_iterator unvectorize(
        const Vector &v, bool minimal=true);
    virtual ostream &print(ostream &out)const;

    // Adding data only updates the upper triangle of xtx_.  Calling
    // reflect() fills the lower triangle as well, if needed.
    void reflect()const;
  private:
    mutable Spd xtx_;
    mutable bool needs_to_reflect_;
    Vector xty_;
    bool xtx_is_fixed_;
    double sumsqy;
    double n_;
    double sumy_;
  };

  template <class Fwd>
  NeRegSuf::NeRegSuf(Fwd b, Fwd e){
    Ptr<RegressionData> dp = *b;
    uint p = dp->xdim();
    xtx_ = Spd(p, 0.0);
    xty_ = Vec(p, 0.0);
    sumsqy = 0.0;
    while(b!=e){
      update(*b);
      ++b;
    }
  }


  //------------------------------------------------------------------
  class RegressionDataPolicy
    : public SufstatDataPolicy<RegressionData, RegSuf>
  {
  public:
    typedef RegressionDataPolicy DataPolicy;
    typedef SufstatDataPolicy<RegressionData, RegSuf> DPBase;

    RegressionDataPolicy(Ptr<RegSuf>);
    RegressionDataPolicy(Ptr<RegSuf>, const DatasetType &d);
    template <class FwdIt>
    RegressionDataPolicy(Ptr<RegSuf>, FwdIt Begin, FwdIt End);

    RegressionDataPolicy(const RegressionDataPolicy &);
    RegressionDataPolicy * clone()const=0;
    RegressionDataPolicy & operator=(const RegressionDataPolicy &);

  };
  template <class Fwd>
  RegressionDataPolicy::RegressionDataPolicy(Ptr<RegSuf> s, Fwd b, Fwd e)
    : DPBase(s,b,e)
  {}

  //------------------------------------------------------------------

  class RegressionModel
    : public GlmModel,
      public ParamPolicy_2<GlmCoefs, UnivParams>,
      public RegressionDataPolicy,
      public ConjugatePriorPolicy<RegressionConjSampler>,
      public NumOptModel,
      public EmMixtureComponent
  {
 public:
    RegressionModel(unsigned int p);
    RegressionModel(const Vector &b, double Sigma);
    RegressionModel(const Matrix &X, const Vector &y);
    RegressionModel(const DatasetType &d, bool include_all_variables = true);
    RegressionModel(const RegressionModel &rhs);
    RegressionModel * clone()const;

    uint nvars()const;  // number of included variables, inc. intercept
    uint nvars_possible()const;  // number of potential variables, inc. intercept

    //---- parameters ----
    GlmCoefs & coef();
    const GlmCoefs & coef()const;
    Ptr<GlmCoefs> coef_prm();
    const Ptr<GlmCoefs> coef_prm()const;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm()const;

    void set_sigsq(double s2);

    double sigsq()const;
    double sigma()const;

    //---- simulate regression data  ---
    virtual RegressionData * simdat()const;
    virtual RegressionData * simdat(const Vector &X)const;
    Vector simulate_fake_x()const;  // no intercept

    //---- estimation ---
    Spd xtx(const Selector &inc)const;
    Vector xty(const Selector &inc)const;
    Spd xtx()const;      // adjusts for covariate inclusion-
    Vector xty()const;      // exclusion, and includes weights,
    double yty()const;   // if used

    void make_X_y(Mat &X, Vector &y)const;

    //--- probability calculations ----
    virtual void mle();
    virtual double Loglike(Vector &g, Mat &h, uint nd)const;
    virtual double pdf(dPtr, bool)const;
    virtual double pdf(const Data *, bool)const;

    // The log likelihood when beta is empty (i.e. all coefficients,
    // including the intercept, are zero).
    double empty_loglike(Vector &g, Mat &h, uint nd)const;

    // If the model was formed using the QR decomposition, switch to
    // using the normal equations.  The normal equations are
    // computationally more efficient when doing variable selection or
    // when the data is changing between MCMC iterations (as in finite
    // mixtures).
    void use_normal_equations();

    void add_mixture_data(Ptr<Data>, double prob);

    void set_conjugate_prior(Ptr<MvnGivenXandSigma>, Ptr<GammaModel>);
    void set_conjugate_prior(Ptr<RegressionConjSampler>);

    //--- diagnostics ---
    AnovaTable anova()const{return suf()->anova();}
  };
  //------------------------------------------------------------

}// ends namespace BOOM
#endif
