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
#include <Models/ProductDirichletModel.hpp>
#include <cmath>
#include <distributions.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>
#include <LinAlg/Matrix.hpp>

namespace BOOM{
  typedef ProductDirichletSuf PDS;
  PDS::ProductDirichletSuf(uint p)
    : sumlog_(p,p,0.0)
  {}

  PDS::ProductDirichletSuf(const PDS &rhs)
    : Sufstat(rhs),
      SufstatDetails<MatrixData>(rhs),
      sumlog_(rhs.sumlog_)
  {}

  PDS * PDS::clone()const{return new PDS(*this);}

  const Mat & PDS::sumlog()const{return sumlog_;}
  double PDS::n()const{return n_;}

  void PDS::Update(const MatrixData &d){
    sumlog_ += log(d.value());
    ++n_;
  }

  void PDS::clear(){
    sumlog_=0;
    n_=0;
  }

  void PDS::combine(Ptr<PDS> s){
    sumlog_ += s->sumlog_;
    n_ += s->n_;
  }

  void PDS::combine(const PDS & s){
    sumlog_ += s.sumlog_;
    n_ += s.n_;
  }

  ProductDirichletSuf * PDS::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this,s); }

  Vec PDS::vectorize(bool)const{
    Vec ans(sumlog_.begin(), sumlog_.end());
    ans.push_back(n_);
    return ans;
  }

  Vec::const_iterator PDS::unvectorize(Vec::const_iterator &v, bool){
    uint dim = sumlog_.nrow();
    Mat tmp(v, v + dim*dim, dim, dim);
    v+= dim*dim;
    sumlog_ = tmp;
    n_ = *v; ++v;
    return v;
  }

  Vec::const_iterator PDS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream &PDS::print(ostream &out)const{
    return out << n_ << endl << sumlog_;
  }
  //============================================================

  typedef ProductDirichletModel PDM;

  PDM::ProductDirichletModel(uint p)
    : ParamPolicy(new MatrixParams(p,p, 1.0)),
      DataPolicy(new PDS(p)),
      PriorPolicy()
  {}

  PDM::ProductDirichletModel(const Mat &N)
    : ParamPolicy(new MatrixParams(N)),
      DataPolicy(new PDS(N.nrow())),
      PriorPolicy()
  {}

  PDM::ProductDirichletModel(const Vec &wgt, const Mat &Pi)
    : ParamPolicy(new MatrixParams(Pi)),
      DataPolicy(new PDS(wgt.size())),
      PriorPolicy()
  {
    Spd W(wgt.size());
    W.set_diag(wgt);
    set_Nu(W*Nu());
  }

  PDM::ProductDirichletModel(const PDM &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      //      DiffVectorModel(rhs),
      dLoglikeModel(rhs)
  {}

  PDM * PDM::clone()const{return new PDM(*this);}

  uint PDM::dim()const{ return Nu().nrow();  }

  Ptr<MatrixParams> PDM::Nu_prm(){
    return ParamPolicy::prm();}
  const Ptr<MatrixParams> PDM::Nu_prm()const{
    return ParamPolicy::prm();}
  const Mat & PDM::Nu()const{return Nu_prm()->value();}

  void PDM::set_Nu(const Mat &Nu){
    Nu_prm()->set(Nu);}

  double PDM::pdf(Ptr<Data> dp, bool logscale)const{
    return pdf(DAT(dp)->value(), logscale); }

  double PDM::pdf(const Mat &Pi, bool logscale)const{
    double ans(0);
    for(uint i=0; i<Pi.nrow(); ++i){
      ans += ddirichlet(Pi.row(i), Nu().row(1), true);
    }
    return logscale ? ans : exp(ans);
  }

  double PDM::loglike()const{
    const Mat & Nu(this->Nu());
    const Mat & sumlog(suf()->sumlog());
    double n=  suf()->n();

    double ans=0;
    for(uint i=0; i<nrow(Nu); ++i)
      ans += dirichlet_loglike(Nu.row(i), 0, 0, sumlog.row(i), n);
    return ans;
  }

  double PDM::dloglike(Vec &g)const{
    const Mat & Nu(this->Nu());
    const Mat & sumlog(suf()->sumlog());
    double n=  suf()->n();

    uint nr = nrow(Nu);
    Mat G(nr,nr);
    Vec g_row(nr);

    double ans=0;
    for(uint i=0; i<nrow(Nu); ++i){
      ans += dirichlet_loglike(Nu.row(i), &g_row, 0, sumlog.row(i), n);
      G.row(i) = g_row;
    }
    G = G.t();
    g.assign(G.begin(), G.end());

    // need to check that g is vectorized in the right way..  virtual
    // functions might expect columns instead of rows.
    return ans;
  }

  Mat PDM::sim()const{
    uint d = dim();
    Mat ans(d,d);
    for(uint i=0; i<d; ++i) ans.row(i) = rdirichlet(Nu().row(i));
    return ans;
  }

}
