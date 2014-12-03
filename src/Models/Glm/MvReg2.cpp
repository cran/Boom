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
#include <Models/Glm/MvReg2.hpp>
#include <boost/bind.hpp>
#include <distributions.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  //======================================================================

  uint MvRegSuf::xdim()const{return xtx().nrow();}
  uint MvRegSuf::ydim()const{return yty().nrow();}

  //======================================================================
  typedef NeMvRegSuf NS;

  NS::NeMvRegSuf(uint xdim, uint ydim)
    : yty_(ydim),
      xtx_(xdim),
      xty_(xdim,ydim),
      n_(0)
  {}

  NS::NeMvRegSuf(const Mat &X, const Mat &Y)
    : yty_(Y.ncol()),
      xtx_(X.ncol()),
      xty_(X.ncol(), Y.ncol()),
      n_(0)
  {
    QR qr(X);
    Mat R = qr.getR();
    xtx_.add_inner(R);

    QR qry(Y);
    yty_.add_inner(qry.getR());

    xty_ = qr.getQ().Tmult(Y);
    xty_ = R.Tmult(xty_);

  }

  NS::NeMvRegSuf(const NS &rhs)
    : Sufstat(rhs),
      MvRegSuf(rhs),
      SufTraits(rhs),
      yty_(rhs.yty_),
      xtx_(rhs.xtx_),
      xty_(rhs.xty_),
      n_(rhs.n_)
  {}

  NS * NS::clone()const{return new NS(*this);}

  void NS::Update(const MvRegData &d){
    const Vec &y(d.y());
    const Vec &x(d.x());
    double w = d.weight();
    update_raw_data(y,x,w);
  }


  void NS::update_raw_data(const Vec &y, const Vec &x, double w){
    ++n_;
    sumw_+=w;
    xtx_.add_outer(x,w);
    xty_.add_outer(x,y,w);
    yty_.add_outer(y,w);
  }

  Mat NS::beta_hat()const{ return xtx_.solve(xty_);}


  Spd NS::SSE(const Mat &B)const{
    Spd ans = yty();
    ans.add_inner2(B, xty(), -1);
    ans+= sandwich(B.t(), xtx());
    return ans;
  }

  void NeMvRegSuf::clear(){
    yty_ = 0;
    xtx_ = 0;
    xty_ = 0;
    n_=0;
  }

  const Spd & NS::yty()const{return yty_;}
  const Spd & NS::xtx()const{return xtx_;}
  const Mat & NS::xty()const{return xty_;}
  double NS::n()const{return n_;}
  double NS::sumw()const{return sumw_;}

  void NS::combine(Ptr<MvRegSuf> sp){
    Ptr<NS> s(sp.dcast<NS>());
    xty_ += s->xty_;
    xtx_ += s->xtx_;
    yty_ += s->yty_;
    sumw_ += s->sumw_;
    n_ +=   s->n_;
  }

  void NS::combine(const MvRegSuf  & sp){
    const NS & s(dynamic_cast<const NS &>(sp));
    xty_ += s.xty_;
    xtx_ += s.xtx_;
    yty_ += s.yty_;
    sumw_ += s.sumw_;
    n_ +=   s.n_;
  }

  Vec NS::vectorize(bool minimal)const{
    Vec ans = yty_.vectorize(minimal);
    ans.concat(xtx_.vectorize(minimal));
    Vec tmp(xty_.begin(), xty_.end());
    ans.concat(tmp);
    ans.push_back(sumw_);
    ans.push_back(n_);
    return ans;
  }

  NeMvRegSuf * NS::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this,s); }

  Vec::const_iterator NS::unvectorize(Vec::const_iterator &v,
                                      bool minimal){
    yty_.unvectorize(v,minimal);
    xtx_.unvectorize(v,minimal);
    uint xdim = xtx_.nrow();
    uint ydim = yty_.nrow();
    Mat tmp(v, v+xdim*ydim, xdim, ydim);
    v+=xdim*ydim;
    sumw_ = *v; ++v;
    n_    = *v; ++v;
    return v;
  }

  Vec::const_iterator NS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & NS::print(ostream &out)const{
    out << "yty_ = " << yty_ << endl
        << "xty_ = " << xty_ << endl
        << "xtx_ = " << endl << xtx_;
    return out;
  }

  //======================================================================

  typedef QrMvRegSuf QS;
  QS::QrMvRegSuf(const Mat &X, const Mat &Y, MvReg *Owner)
    : qr(X),
      owner(Owner),
      current(false),
      yty_(Y.ncol()),
      xtx_(X.ncol()),
      xty_(X.ncol(), Y.ncol()),
      n_(0)
  {
    refresh(X,Y);
    current=true;
  }

  QS::QrMvRegSuf(const Mat &X, const Mat &Y, const Vec &W, MvReg *Owner)
    : qr(X),
      owner(Owner),
      current(false),
      yty_(Y.ncol()),
      xtx_(X.ncol()),
      xty_(X.ncol(), Y.ncol()),
      n_(0)
  {
    refresh(X,Y,W);
    current=true;
  }

  void QS::combine(Ptr<MvRegSuf>){
    throw_exception<std::runtime_error>("cannot combine QrMvRegSuf");
  }

  void QS::combine(const MvRegSuf &){
    throw_exception<std::runtime_error>("cannot combine QrMvRegSuf");
  }

  QrMvRegSuf * QS::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this,s); }


  Vec QS::vectorize(bool)const{
    throw_exception<std::runtime_error>("cannot vectorize QrMvRegSuf");
    return Vec(1,0.0);
  }

  Vec::const_iterator QS::unvectorize(Vec::const_iterator &v,
                                      bool){
    throw_exception<std::runtime_error>("cannot unvectorize QrMvRegSuf");
    return v;
  }

  Vec::const_iterator QS::unvectorize(const Vec &v, bool minimal){
    throw_exception<std::runtime_error>("cannot unvectorize QrMvRegSuf");
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & QS::print(ostream &out)const{
    out << "yty_ = " << yty_ << endl
        << "xty_ = " << xty_ << endl
        << "xtx_ = " << endl << xtx_;
    return out;
  }

  QS * QS::clone()const{return new QS(*this);}

  const Spd & QS::xtx()const{
    if(!current) refresh();
    return xtx_;
  }
  const Spd & QS::yty()const{
    if(!current) refresh();
    return yty_;
  }
  const Mat & QS::xty()const{
    if(!current) refresh();
    return xty_;
  }
  double QS::n()const{
    if(!current) refresh();
    return n_;}
  double QS::sumw()const{
    if(!current) refresh();
    return sumw_;}


  void QS::refresh(const Mat &X, const Mat &Y)const{
    y_ = Y;
    qr.decompose(X);
    Mat R(qr.getR());
    xtx_ = 0;
    xtx_.add_inner(R);

    QR qry(Y);
    xty_ = 0;
    yty_.add_inner(qry.getR());

    n_ = X.nrow();

    xty_ = qr.getQ().Tmult(Y);
    xty_ = R.Tmult(xty_);
    current = true;
  }

  void QS::refresh(const Mat &X, const Mat &Y, const Vec &w )const{
    y_ = Y;
    Mat x_(X);
    uint nr = X.nrow();
    sumw_=0;
    for(uint i=0; i<nr; ++i){
      sumw_+= w[i];
      double rootw = sqrt(w[i]);
      y_.row(i) *= rootw;
      x_.row(i) *= rootw;
    }
    qr.decompose(x_);
    Mat R(qr.getR());
    xtx_ = 0;
    xtx_.add_inner(R);

    QR qry(y_);
    xty_ = 0;
    yty_.add_inner(qry.getR());

    n_ = X.nrow();

    xty_ = qr.getQ().Tmult(y_);
    xty_ = R.Tmult(xty_);
    current = true;
  }

  void QS::refresh(const std::vector<Ptr<MvRegData> > &dv)const{
    Ptr<MvRegData> dp = dv[0];
    uint n = dv.size();
    const Vec &x0(dp->x());
    const Vec &y0(dp->y());

    uint nx = x0.size();
    uint ny = y0.size();
    sumw_ = 0;
    Mat X(n, nx);
    Mat Y(n, ny);
    for(uint i=0; i<n; ++i){
      dp = dv[i];
      double w = dp->weight();
      sumw_ += w;
      if(w==1.0){
	X.set_row(i, dp->x());
	Y.set_row(i, dp->y());
      }else{
	double rootw = sqrt(w);
	X.set_row(i,dp->x() * rootw);
	Y.set_row(i,dp->y() * rootw);
      }
    }
    refresh(X,Y);
  }

  void QS::refresh()const{
    refresh(owner->dat());
  }

  void QS::clear(){
    current=false;
    n_=0;
    xtx_=0;
    xty_=0;
    yty_=0;
  }


  void QS::Update(const MvRegData &){
    current = false;
  }


  Mat QS::beta_hat()const{
    Mat ans = qr.getQ().Tmult(y_);
    ans = qr.Rsolve(ans);
    return ans;
  }

  Spd QS::SSE(const Mat &B)const{
    Mat RB = qr.getR() * B;
    Spd ans = yty();
    ans.add_inner(RB);

    Mat Qty = qr.getQ().Tmult(y_);
    Mat tmp = RB.Tmult(Qty);
    ans.add_inner2(RB, Qty, -1.0);

    return ans;
  }
  //======================================================================

  MvReg::MvReg(uint xdim, uint ydim)
    : ParamPolicy( new MatrixParams(xdim,ydim), new SpdParams(ydim)),
      DataPolicy(new NeMvRegSuf(xdim, ydim)),
      PriorPolicy(),
      LoglikeModel()
  {
  }

  MvReg::MvReg(const Mat &X, const Mat &Y)
    : ParamPolicy(),
      DataPolicy(new QrMvRegSuf(X, Y, this)),
      PriorPolicy(),
      LoglikeModel()
  {
    uint nx = X.ncol();
    uint ny = Y.ncol();
    set_params( new MatrixParams(nx,ny), new SpdParams(ny));
    mle();
  }

  MvReg::MvReg(const Mat &B, const Spd & V)
    : ParamPolicy(new MatrixParams(B), new SpdParams(V)),
      DataPolicy(new NeMvRegSuf(B.nrow(), B.ncol())),
      PriorPolicy(),
      LoglikeModel()
  {
  }

  MvReg::MvReg(const MvReg &rhs)
    : Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      LoglikeModel(rhs)
  {
  }

  MvReg * MvReg::clone()const{return new MvReg(*this);}

  uint MvReg::xdim()const{return Beta().nrow();}
  uint MvReg::ydim()const{return Beta().ncol();}

  const Mat & MvReg::Beta()const{ return Beta_prm()->value(); }
  const Spd & MvReg::Sigma()const{ return Sigma_prm()->var();}
  const Spd & MvReg::Siginv()const{return Sigma_prm()->ivar();}
  double MvReg::ldsi()const{return Sigma_prm()->ldsi();}

  Ptr<MatrixParams> MvReg::Beta_prm(){return prm1();}
  const Ptr<MatrixParams> MvReg::Beta_prm()const{return prm1();}
  Ptr<SpdParams> MvReg::Sigma_prm(){return prm2();}
  const Ptr<SpdParams> MvReg::Sigma_prm()const{return prm2();}

  void MvReg::set_Beta(const Mat &B){
    Beta_prm()->set(B);
  }

  void MvReg::set_Sigma(const Spd &V){
    Sigma_prm()->set_var(V); }

  void MvReg::set_Siginv(const Spd &iV){
    Sigma_prm()->set_ivar(iV); }

  void MvReg::mle(){
    set_Beta(suf()->beta_hat());
    set_Sigma(suf()->SSE(Beta())/suf()->n());
  }

  double MvReg::loglike(const Vector &beta_siginv)const{
    const double log2pi = 1.83787706641;

    Matrix Beta(xdim(), ydim());
    Vector::const_iterator it = beta_siginv.cbegin();
    std::copy(it, it + Beta.size(), Beta.begin());
    it += Beta.size();
    SpdMatrix siginv(ydim());
    siginv.unvectorize(it, true);

    const Spd & yty(suf()->yty());
    const Mat & xty(suf()->xty());
    const Spd & xtx(suf()->xtx());

    double qform = traceAB(siginv, yty) - 2* traceAB(xty.multT(Beta), siginv)
      + traceAB(sandwich(Beta, siginv), xtx);

    double ldsi = siginv.logdet();
    double n = suf()->n();
    double ans = log2pi*n/2 + ldsi*n/2 - .5*qform;
    return ans;
  }

  double MvReg::pdf(dPtr dptr, bool logscale)const{
    Ptr<MvRegData> dp = DAT(dptr);
    Vec mu = predict(dp->x());
    return dmvn(dp->y(), mu, Siginv(), ldsi(), logscale);
  }

  Vec MvReg::predict(const Vec &x)const{ return x* Beta(); }

  MvRegData * MvReg::simdat()const{
    Vec x = simulate_fake_x();
    return simdat(x);
  }

  MvRegData * MvReg::simdat(const Vec &x)const{
    Vec mu = predict(x);
    Vec y = rmvn(mu, Sigma());
    return new MvRegData(y,x);
  }

  Vec MvReg::simulate_fake_x()const{
    uint p = xdim();
    Vec x(p, 1.0);
    for(uint i=1; i<p; ++i) x[i]=rnorm();
    return x;
  }

}
