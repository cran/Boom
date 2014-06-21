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

#ifndef GLM_MODEL_H
#define GLM_MODEL_H
#include <BOOM.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/Glm/GlmCoefs.hpp>
#include <LinAlg/Selector.hpp>
#include <LinAlg/VectorView.hpp>

namespace BOOM{
  template <class DAT>
  class GlmData : virtual public Data{  // intercept stored explicitly.
    Ptr<VectorData> x_;                 // makes life simple.
    Ptr<DAT> y_;
  public:
    typedef typename DAT::value_type value_type;
    typedef const value_type const_value_type;

    // copies data in X
    GlmData(const value_type &y, const Vector &X, bool add_intercept = false);

    // size is p or p+1
    GlmData(Ptr<DAT> yp, const Vector &X, bool add_intercept =false);
    GlmData(const value_type &y, uint p);  // copies data in X
    GlmData(Ptr<DAT> yp, uint p);         // size is p or p+1
    GlmData(Ptr<DAT> yp, Ptr<VectorData> xp);
    GlmData(const GlmData &);                         // value semantics
    GlmData * clone()const;

    // required virtual functions
    uint size(bool minimal=true)const;   // includes intercept, if present
    virtual ostream & display(ostream &)const;
    //    virtual istream & read(istream &);

    uint xdim()const;
    const Vector &x()const;
    void set_x(const Vector &X, bool allow_any=false);
    const value_type &y()const;
    void set_y(const value_type &Y);
    virtual double weight()const{return 1.0;}

    Ptr<VectorData> Xptr(){return x_;}
    Ptr<DAT> Yptr(){return y_;}
  };

  //------------------------------------------------------------
  template <class D>
  class WeightedGlmData : virtual public Data{
  public:
    typedef typename GlmData<D>::value_type value_type;

    WeightedGlmData(Ptr<GlmData<D> > dp, double W)
      : dat_(dp), w_(new DoubleData(W)) {}

    WeightedGlmData(Ptr<GlmData<D> > dp, Ptr<DoubleData> W)
      : dat_(dp), w_(W) {}

    WeightedGlmData(const WeightedGlmData &rhs)
      : Data(rhs), dat_(rhs.dat_->clone()), w_(rhs.w_->clone()){}

    WeightedGlmData * clone()const{return new WeightedGlmData(*this);}

    uint xdim() const{ return dat_->xdim(); }
    ostream & display(ostream & out)const{
      w_->display(out); dat_->display(out); return out;}

    //    istream & read(istream & in){w_->read(in); dat_->read(in); return in;}

    const Vector &x()const{return dat_->x();}
    void set_x(const Vector &X, bool allow_any=false){dat_->set_x(X,allow_any);}
    const value_type &y()const{return dat_->y();}
    void set_y(const value_type &Y){dat_->set_y(Y);}
    virtual double weight()const{return w_->value();}
    void set_weight(double W){w_->set(W);}

    Ptr<DoubleData> WeightPtr(){return w_;}
  private:
    Ptr<GlmData<D> > dat_;
    Ptr<DoubleData> w_;
  };


  //============================================================
  class OrdinalData;

  typedef GlmData<DoubleData> RegressionData;
  typedef GlmData<VectorData> MvRegData;
  typedef GlmData<BinaryData> BinaryRegressionData;
  typedef GlmData<OrdinalData> OrdinalRegressionData;
  typedef WeightedGlmData<DoubleData> WeightedRegressionData;
  typedef WeightedGlmData<VectorData> WeightedMvRegData;

  class GlmModel: virtual public Model{
  public:
    GlmModel();
    GlmModel(const GlmModel &rhs);
    virtual GlmModel *clone()const=0;

    virtual GlmCoefs & coef() = 0;
    virtual const GlmCoefs & coef()const = 0;
    virtual Ptr<GlmCoefs> coef_prm()=0;
    virtual const Ptr<GlmCoefs> coef_prm()const=0;

    uint xdim()const;
    //---- model selection ----
    void add_all();
    void drop_all();
    void drop_all_but_intercept();
    void add(uint p);
    void drop(uint p);
    void flip(uint p);
    const Selector  & inc()const;
    bool inc(uint p)const;

    Vector included_coefficients()const;
    void set_included_coefficients(const Vector &b);

    // Set the included coefficients to those
    void set_included_coefficients(const Vector &beta,
                                   const Selector &inc);

    //---------------
    virtual const Vector & Beta()const; // reports 0 for excluded positions

    // Set the full vector of regression coefficients to Beta.
    void set_Beta(const Vector & Beta);

    double Beta(uint I)const;        // I indexes possible covariates

    virtual double predict(const Vector &x)const;
    virtual double predict(const VectorView &x)const;
    virtual double predict(const ConstVectorView &x)const;
  };

  //============================================================
  template <class D>
  GlmData<D>::GlmData(const value_type &Y, const Vector &X, bool icpt)
    : x_(new VectorData(icpt? concat(1.0, X) : X)),
      y_(new D(Y))
  {}

  template <class D>
  GlmData<D>::GlmData(Ptr<D> Y, const Vector &X, bool icpt)
    : x_(new VectorData(icpt? concat(1.0, X) : X)),
      y_(Y->clone())
  {}

  template <class D>
  GlmData<D>::GlmData(const value_type &Y, uint p)
    : x_(new VectorData(p)),
    y_(new D(Y))
  {}

  template <class D>
  GlmData<D>::GlmData(Ptr<D> Y, uint p)
    : x_(new VectorData(p)),
      y_(Y->clone())
  {}

  template <class D>
  GlmData<D>::GlmData(Ptr<D> Y, Ptr<VectorData> X)
    : x_(X),
      y_(Y)
  {}

  template<class D>
  GlmData<D>::GlmData(const GlmData &rhs)
    : Data(rhs),
      x_(rhs.x_->clone()),
      y_(rhs.y_->clone())
  {}

  template<class D>
  GlmData<D> * GlmData<D>::clone()const{
    return new GlmData(*this);}

  template<class D>
  ostream & GlmData<D>::display(ostream &out)const{
    y_->display(out);
    out << " ";
    x_->display(out);
    return out;
  }

  template <class D>
  uint GlmData<D>::xdim()const{ return x_->dim();}

  template<class D>
  const Vector &GlmData<D>::x()const{ return x_->value();}

  void incompatible_x(const Vector &X, const Vector & x);

  template<class D>
  void GlmData<D>::set_x(const Vector &X, bool allow_any){
    if(allow_any || x_->dim()== X.size()) x_->set(X);
    else if(x_->dim()== 1 + X.size()) x_->set(concat(1.0, X));
    else incompatible_x(X,x());
    signal();
  }

  template<class D>
  const typename D::value_type & GlmData<D>::y()const{
    return y_->value();}

  template<class D>
  void GlmData<D>::set_y(const value_type &Y){
    y_->set(Y);}

  //============================================================

}
 #endif // GLM_MODEL_H
