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

#ifndef BOOM_WEIGHTED_REGRESSION_MODEL_HPP
#define BOOM_WEIGHTED_REGRESSION_MODEL_HPP

#include <Models/Glm/RegressionModel.hpp>
#include <Models/Glm/Glm.hpp>

namespace BOOM{

  //------------------------------------------------------------

  class WeightedRegSuf
    : public SufstatDetails<WeightedRegressionData>
  {
  private:
    mutable Spd xtwx_;
    Vec xtwy_;
    double n_;  // xtx_(0,0) is the sum of the weights,
    double yt_w_y_;
    double sumlogw_;
    mutable bool sym_;
    void setup_mat(uint p);
    void make_symmetric()const;
  public:
    typedef WeightedRegressionData data_type;
    typedef std::vector<Ptr<WeightedRegressionData> > dataset_type;
    typedef Ptr<dataset_type, false> dsetPtr;

    WeightedRegSuf(int p); // dimension of beta
    WeightedRegSuf(const Mat &X, const Vec &y); // w implicitly 1.0
    WeightedRegSuf(const Mat &X, const Vec &y, const Vec &w);
    WeightedRegSuf(const dsetPtr &dat);
    WeightedRegSuf(const WeightedRegSuf &rhs);  // value semantics

    WeightedRegSuf * clone()const;

    virtual void reweight(const Mat &X, const Vec &y, const Vec &w);
    virtual void reweight(const dsetPtr &dat);

    //    virtual void Update(const RegressionData &);
    virtual void Update(const WeightedRegressionData &);
    void add_data(const Vec &x, double y, double w);
    virtual void clear();
    virtual uint size()const;  // dimension of beta
    virtual double yty()const;              // Y^t W Y
    virtual Vec xty()const;                 // X^T W Y
    virtual Spd xtx()const;                 // X^T W X
    virtual Vec xty(const Selector &)const;  // X^T W Y
    virtual Spd xtx(const Selector &)const;  // X^T W X
    virtual Vec beta_hat()const;            // WLS estimate
    virtual double SSE()const;   //
    virtual double SST()const;   // weighted sum of squares
    virtual double ybar()const;  // weighted average
    virtual double n()const;
    virtual double sumw()const; // sum of weights
    virtual double sumlogw()const; // sum of weights
    virtual ostream & print(ostream &out)const;
    void combine(Ptr<WeightedRegSuf>);
    void combine(const WeightedRegSuf &);
    WeightedRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                            bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
                                            bool minimal=true);
  };

  inline ostream & operator<<(ostream & out, const WeightedRegSuf &s){
    return s.print(out);}

  //------------------------------------------------------------

  class WeightedRegressionModel
    : public ParamPolicy_2<GlmCoefs, UnivParams>,
      public SufstatDataPolicy<WeightedRegressionData, WeightedRegSuf>,
      public PriorPolicy,
      public GlmModel,
      public NumOptModel
  {
  public:
    typedef WeightedRegressionData data_type;
    typedef WeightedRegSuf suf_type;

    WeightedRegressionModel(uint p);
    WeightedRegressionModel(const Vec &b, double Sigma);
    WeightedRegressionModel(const WeightedRegressionModel &rhs);
    WeightedRegressionModel(const Mat &X, const Vec &y);
    WeightedRegressionModel(const Mat &X, const Vec &y, const Vec &w);
    WeightedRegressionModel(const DatasetType &d, bool all=true);
    WeightedRegressionModel * clone()const;

    virtual GlmCoefs & coef();
    virtual const GlmCoefs & coef()const;
    virtual Ptr<GlmCoefs> coef_prm();
    virtual const Ptr<GlmCoefs> coef_prm()const;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm()const;

    // beta() and Beta() inherited from GLM;
    //    void set_beta(const Vec &b);
    void set_sigsq(double s2);

    const double & sigsq()const;
    double sigma()const;

    void mle();

    double Loglike(Vec &g, Mat &h, uint nd)const;
    double pdf(dPtr, bool)const;
    double pdf(Ptr<data_type>, bool)const;

  };

}

#endif // BOOM_WEIGHTED_REGRESSION_MODEL_HPP
