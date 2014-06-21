/*
  Copyright (C) 2007 Steven L. Scott

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
#include <Models/Glm/GlmMvnPrior.hpp>
#include <LinAlg/Cholesky.hpp>
#include <distributions.hpp>
#include <LinAlg/Givens.hpp>
#include <cpputil/seq.hpp>
#include <cpputil/nyi.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  typedef GlmMvnSuf GMS;

  GMS::GlmMvnSuf(uint p)
    : bbt_(p),
      ggt_(p),
      bgt_(p,p),
      vnobs_(p),
      nobs_(0),
      b(p)
  {}

  GMS::GlmMvnSuf(const std::vector<Ptr<GlmCoefs> > &d)
    : bbt_(d[0]->dim()),
      ggt_(bbt_),
      bgt_(bbt_),
      vnobs_(bbt_.nrow()),
      nobs_(0),
      b(vnobs_),
      sym_(false)
  {
    uint n = d.size();
    for(uint i=0; i<n; ++i) Update(*d[i]);
  }

  GMS * GMS::clone()const{return new GMS(*this);}

  void GMS::Update(const GlmCoefs &beta){
    ++nobs_;
    b  = beta.Beta();
    const Selector &inc(beta.inc());
    gam = inc.vec();
    bbt_.add_outer(b, 1.0, false);
    bgt_.add_outer(b, gam);
    ggt_.add_outer(gam, 1.0, false);
    vnobs_ += gam;
    sym_ = false;
  }

  void GMS::make_symmetric()const{
    if(sym_) return;
    sym_= true;
    bbt_.reflect();
    ggt_.reflect();
  }

  void GMS::clear(){
    bbt_ = 0;
    ggt_ = 0;
    bgt_ = 0;
    nobs_ = 0;
    vnobs_ = 0;
  }

  const Vec & GMS::vnobs()const{ return vnobs_;}
  uint GMS::nobs()const{return nobs_;}
  const Spd & GMS::GTG()const{
    make_symmetric();
    return ggt_;
  }
  const Mat & GMS::BTG()const{ return bgt_;}

  Spd GMS::center_sumsq(const Vec &b)const{
    make_symmetric();
    Spd ans = bbt_;
    Mat tmp = bgt_ * diag(b);
    ans -= tmp + tmp.t();
    ans += el_mult(GTG(), outer(b));
    return ans;
  }

  void GMS::combine(Ptr<GMS> s){
    bbt_ += s->bbt_;
    ggt_ += s->ggt_;
    bgt_ += s->bgt_;
    vnobs_ += s->vnobs_;
    nobs_ += s->nobs_;
    sym_ = sym_ && s->sym_;
  }

  void GMS::combine(const GMS & s){
    bbt_ += s.bbt_;
    ggt_ += s.ggt_;
    bgt_ += s.bgt_;
    vnobs_ += s.vnobs_;
    nobs_ += s.nobs_;
    sym_ = sym_ && s.sym_;
  }

  GlmMvnSuf * GMS::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this,s);}

  Vec GMS::vectorize(bool minimal)const{
    Vec ans = bbt_.vectorize(minimal);
    ans.concat(ggt_.vectorize(minimal));
    Vec tmp(bgt_.begin(), bgt_.end());
    ans.concat(tmp);
    ans.concat(vnobs_);
    ans.push_back(nobs_);
    return ans;
  }

  Vec::const_iterator GMS::unvectorize(Vec::const_iterator &v,
                                       bool minimal){
    bbt_.unvectorize(v, minimal);
    ggt_.unvectorize(v,minimal);
    uint dim = bbt_.nrow();
    Mat tmp(v, v + dim*dim, dim, dim);
    v+= dim*dim;
    bgt_ = tmp;
    vnobs_.assign(v, v+dim);
    v+=dim;
    nobs_ = lround(*v);
    ++v;
    sym_ = false;
    return v;
  }

  Vec::const_iterator GMS::unvectorize(const Vec &v,
                                       bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & GMS::print(ostream &out)const{
    out << "bbt_ = " << endl << bbt_
        << "ggt_ = " << endl << ggt_
        << "bgt_ = " << bgt_ << endl
        << "vnobs_ = " << vnobs_ << endl
        << "nobs_ = " << nobs_ << endl
        << "b = " << b << endl
        << "gam = " << gam << endl
        << "sym_ = " << sym_ << endl;
    return out;
  }

  //______________________________________________________________________

  typedef GlmMvnPrior GMP;

  GMP::GlmMvnPrior(uint p, double mu, double sig)
    : Base(p,mu,sig),
      DataPolicy(new GlmMvnSuf(p))
  {}

  GMP::GlmMvnPrior(const Vec &mean, const Spd &v, bool ivar)
    : Base(mean,v,ivar),
      DataPolicy(new GlmMvnSuf(mean.size()))
  { }

  GMP::GlmMvnPrior(Ptr<VectorParams> mu, Ptr<SpdParams> Sigma)
    : Base(mu, Sigma),
      DataPolicy(new GlmMvnSuf(mu->size()))
  { }

  GMP::GlmMvnPrior(const std::vector<Ptr<GlmCoefs> > &dat)
    : Base(dat[0]->nvars_possible()),
      DataPolicy(new GlmMvnSuf(dat), dat)
  {}

  GMP::GlmMvnPrior(const GMP & rhs)
    : Model(rhs),
      VectorModel(rhs),
      MLE_Model(rhs),
      Base(rhs),
      LoglikeModel(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs)
  {}

  GMP * GMP::clone()const{return new GMP(*this);}

  //------------------------------------------------------------
  void GMP::mle(){
    nyi("GlmMvnPrior::mle");
  }
  //------------------------------------------------------------

  double GMP::loglike()const{
    Ptr<GlmMvnSuf> s(suf());
    double qform = -0.5 * traceAB(siginv(), s->center_sumsq(mu()));

    const std::vector<Ptr<GlmCoefs> > & dat(GMP::dat());
    uint n = dat.size();
    Mat LT = siginv_chol().t();
    double logdet = 0;
    for(uint i=0; i<n; ++i){
      const Selector & inc(dat[i]->inc());
      logdet += eval_logdet(inc,LT);
    }
    // |L| = |siginv|^1/2 because the cholesky and the 1/2 cancel


    const Vec  &nobs(s->vnobs());
    const double minus_half_log_2pi = -0.918938533204673;
    double nc = minus_half_log_2pi * sum(nobs);

    double ans = nc + logdet + qform;

    return ans;
  }


  double GMP::eval_logdet(const Selector &inc, const Mat & LT)const{
    // evaluates the determinant of a subset of an upper triangular
    // matrix LT.
    Mat R = triangulate(LT, inc, false);
    const VectorView d(R.diag());
    double ans=0;
    for(uint k=0; k<d.size(); ++k) ans += log(fabs(d[k]));
    return ans;
  }

  Vec GMP::sim()const{
    return Vec(1,0.0);
  }

  Spd GMP::sumsq()const{ return suf()->center_sumsq(mu()); }

  Vec GMP::simulate(const Selector &inc)const{

    Vec b = inc.select(mu());
    Mat R = siginv_chol().t();
    R = triangulate(R, inc, true);
    uint nc = R.ncol();
    Vec z(nc);
    for(uint i=0; i<nc; ++i) z[i] = rnorm();
    b+= Usolve_inplace(R, z);
    return inc.expand(b);
  }

  double GMP::pdf(const Ptr<GlmCoefs> beta, bool logscale)const{
    const Selector & inc(beta->inc());
    ivar_ = inc.select(siginv());
    wsp_ = inc.select(mu());
    double ans = dmvn(beta->Beta(), wsp_, ivar_, true);
    return logscale ? ans : exp(ans);
  }
  double GMP::pdf(Ptr<Data> dp, bool logscale)const{
    return pdf(DAT(dp),logscale);}

  const Mat & GMP::siginv_chol()const{
    return Sigma_prm()->ivar_chol();
  }

}
