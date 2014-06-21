/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_MVREG_HPP
#define BOOM_MVREG_HPP
#include <Models/Sufstat.hpp>
#include <Models/SpdParams.hpp>
#include <Models/Glm/Glm.hpp>
#include <LinAlg/QR.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <boost/bind.hpp>

namespace BOOM{

  class MvRegSuf : virtual public Sufstat{
  public:
    typedef std::vector<Ptr<MvRegData> > dataset_type;
    typedef Ptr<dataset_type, false> dsetPtr;

    MvRegSuf * clone()const=0;

    uint xdim()const;
    uint ydim()const;
    virtual const Spd & yty()const=0;
    virtual const Mat & xty()const=0;
    virtual const Spd & xtx()const=0;
    virtual double n()const=0;
    virtual double sumw()const=0;

    virtual Spd SSE(const Mat &B)const=0;

    virtual Mat beta_hat()const=0;
    virtual void combine(Ptr<MvRegSuf>)=0;
  };
  //------------------------------------------------------------
  class MvReg;
  class QrMvRegSuf
    : public MvRegSuf,
      public SufstatDetails<MvRegData>
  {
  public:
    QrMvRegSuf(const Mat &X, const Mat &Y, MvReg *);
    QrMvRegSuf(const Mat &X, const Mat &Y, const Vec &w, MvReg *);
    QrMvRegSuf * clone()const;

    virtual void Update(const MvRegData &);
    virtual Mat beta_hat()const;
    virtual Spd SSE(const Mat &B)const;
    virtual void clear();

    virtual const Spd & yty()const;
    virtual const Mat & xty()const;
    virtual const Spd & xtx()const;
    virtual double n()const;
    virtual double sumw()const;

    void refresh(const std::vector<Ptr<MvRegData> > &)const;
    void refresh(const Mat &X, const Mat &Y)const;
    void refresh(const Mat &X, const Mat &Y, const Vec &w)const;
    void refresh()const;
    virtual void combine(Ptr<MvRegSuf>);
    virtual void combine(const MvRegSuf &);
    QrMvRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                            bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
                                            bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    mutable QR qr;
    mutable Mat y_;
    mutable Vec w_;

    MvReg * owner;

    mutable bool current;
    mutable Spd yty_;
    mutable Spd xtx_;
    mutable Mat xty_;
    mutable double n_;
    mutable double sumw_;
  };

  //------------------------------------------------------------
  // Sufficient statistics for the multivariate regression model based
  // on the normal equations.
  class NeMvRegSuf
    : public MvRegSuf,
      public SufstatDetails<MvRegData>
  {
  public:
    // Args:
    //   xdim:  The dimension of the x (predictor) variable.
    //   ydim:  The dimension of the y (response) variable.
    NeMvRegSuf(uint xdim, uint ydim);

    // Args:
    //   X:  The design matrix.
    //   Y:  The matrix of responses.
    NeMvRegSuf(const Mat &X, const Mat &Y);

    // Build an NeMvRegSuf from a sequence of smart or raw pointers to
    // MvRegData.
    template <class Fwd> NeMvRegSuf(Fwd b, Fwd e);

    NeMvRegSuf(const NeMvRegSuf &rhs);
    NeMvRegSuf * clone()const;

    virtual void clear();

    // Add data to the sufficient statistics managed by this object.
    virtual void Update(const MvRegData & data);

    // Add the individual data components to the sufficient statistics
    // managed by this object.
    // Args:
    //   Y:  The response for a single data point.
    //   X:  The predictor for a single data point.
    //   w:  A weight to apply to the data point.
    virtual void update_raw_data(const Vec &Y, const Vec &X, double w=1.0);

    // Returns the least squares estimate of beta given the current
    // sufficient statistics.
    virtual Mat beta_hat()const;

    // Returns the sum of squared errors assuming beta = B.
    virtual Spd SSE(const Mat &B)const;


    virtual const Spd & yty()const;      // sum_i y_i * y_i.transpose()
    virtual const Mat & xty()const;      // sum_i y_i * x_i.transpose()
    virtual const Spd & xtx()const;      // sum_i x_i * x_i.transpose();
    virtual double n()const;             // number of observations
    virtual double sumw()const;          // sum of weights

    // Add the sufficient statistics managed by the argument to *this.
    virtual void combine(Ptr<MvRegSuf>);
    virtual void combine(const MvRegSuf &);
    NeMvRegSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                            bool minimal = true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
                                            bool minimal = true);
    virtual ostream &print(ostream &out)const;
  private:
    Spd yty_;
    Spd xtx_;
    Mat xty_;
    double sumw_;
    double n_;
  };

  template <class Fwd>
  NeMvRegSuf::NeMvRegSuf(Fwd b, Fwd e)
  {
    Ptr<MvRegData> dp =*b;
    const Vec &x(dp->x());
    const Vec &y(dp->y());

    uint xdim= x.size();
    uint ydim= y.size();
    xtx_ = Spd(xdim, 0.0);
    yty_ = Spd(ydim, 0.0);
    xty_ = Mat(xdim, ydim, 0.0);
    n_ = 0;

    while(b!=e){ this->update(*b); ++b; }
  }

  //============================================================
  // Multivariate regression, where both y_i and x_i are vectors.
  class MvReg
    : public ParamPolicy_2<MatrixParams,SpdParams>,
      public SufstatDataPolicy<MvRegData, MvRegSuf>,
      public PriorPolicy,
      public LoglikeModel
  {
  public:
    // Args:
    //   xdim: The dimension of the predictor, including the intercept
    //     (if any).
    //   ydim:  The dimension of the response.
    MvReg(uint xdim, uint ydim);

    // Args:
    //   X:  The design matrix.
    //   Y:  The matrix of responses.  The number of rows must match X.
    MvReg(const Mat &X, const Mat &Y);

    // Args:
    //   B: The matrix of regression coefficients.  The number of rows
    //     defines the dimension of the predictor.  The number of
    //     columns defines the dimension of the response.
    //   Sigma: The residual variance matrix.  Its dimension must
    //     match ncol(B).
    MvReg(const Mat &B, const Spd &Sigma);

    MvReg(const MvReg & rhs);
    MvReg * clone()const;

    // Dimension of the predictor (including the intercept, if any).
    uint xdim()const;

    // Dimension of the response variable.
    uint ydim()const;

    // Matrix of regression coefficients, with xdim() rows, ydim()
    // columns.
    const Mat & Beta()const;
    void set_Beta(const Mat &B);

    // Residual variance matrix.
    const Spd & Sigma()const;
    void set_Sigma(const Spd &V);

    // Matrix inverse of the residual variance matrix;
    const Spd & Siginv()const;
    void set_Siginv(const Spd &iV);

    // log determinant of Siginv().
    double ldsi()const;

    // Access to parameters.
    Ptr<MatrixParams> Beta_prm();
    const Ptr<MatrixParams> Beta_prm()const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm()const;

    //--- estimation and probability calculations
    virtual void mle();
    virtual double loglike()const;
    virtual double pdf(dPtr, bool)const;

    // Returns x * Beta();
    virtual Vec predict(const Vec &x)const;

    //---- simulate MV regression data ---
    virtual MvRegData * simdat()const;
    virtual MvRegData * simdat(const Vec &X)const;

    // no intercept
    Vec simulate_fake_x()const;
  };
}
#endif // BOOM_MVREG_HPP
