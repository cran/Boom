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

#include <Models/DirichletModel.hpp>
#include <cpputil/math_utils.hpp>
#include <LinAlg/Types.hpp>
#include <stdexcept>
#include <sstream>
#include <distributions.hpp>  // for rgamma, lgamma, digamma, etc.
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <cmath>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  //======================================================================
  typedef DirichletSuf DS;

  DS::DirichletSuf(uint S)
    : sumlog_(S, 0.0),
      n_(0)
  { };

  DS::DirichletSuf(const DirichletSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<VectorData>(rhs),
      sumlog_(rhs.sumlog_),
      n_(rhs.n_)
  {}

  DS * DS::clone()const{return new DS(*this);}

  void DS::clear(){ sumlog_=0.0; n_=0.0;}

  void DS::Update(const VectorData &x){
    n_+=1.0;
    sumlog_+= log(x.value()); }

  void DS::add_mixture_data(const Vec &x, double prob){
    n_ += prob;
    sumlog_.axpy(log(x), prob);
  }

  const Vec & DS::sumlog()const{return sumlog_;}
  double DS::n()const{return n_;}

  void DS::combine(Ptr<DS> s){
    sumlog_ += s->sumlog_;
    n_ += s->n_;
  }

  void DS::combine(const DS & s){
    sumlog_ += s.sumlog_;
    n_ += s.n_;
  }

  DirichletSuf * DS::abstract_combine(Sufstat *s){
      return abstract_combine_impl(this, s);}

  Vec DS::vectorize(bool)const{
    Vec ans = sumlog_;
    ans.push_back(n_);
    return ans;
  }

  Vec::const_iterator DS::unvectorize(Vec::const_iterator &v, bool){
    uint dim = sumlog_.size();

    Vec tmp(v, v+dim);
    v+=dim;
    sumlog_ = tmp;
    n_ = *v;
    return ++v;
  }

  Vec::const_iterator DS::unvectorize(const Vec &v, bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & DS::print(ostream &out)const{
    return out << n_ << " " << sumlog_;
  }
  //======================================================================
  typedef DirichletModel DM;
  typedef VectorParams VP;

  DM::DirichletModel(uint S)
    : ParamPolicy(new VP(S,1.0/S)),
      DataPolicy(new DS(S) ),
      PriorPolicy()
    {}

  DM::DirichletModel(uint S, double Nu)
    : ParamPolicy(new VP(S,Nu)),
      DataPolicy(new DS(S) ),
      PriorPolicy()
    {}

  DM::DirichletModel(const Vec &Nu)
    : ParamPolicy(new VP(Nu)),
      DataPolicy(new DS(Nu.size()) ),
      PriorPolicy()
    {}

  DM::DirichletModel(const DirichletModel &rhs)
    : Model(rhs),
      VectorModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      DiffVectorModel(rhs),
      NumOptModel(rhs),
      EmMixtureComponent(rhs)
  {}

  DM *DM::clone() const{return new DirichletModel(*this);}

  Ptr<VectorParams> DM::Nu(){return ParamPolicy::prm();}
  const Ptr<VectorParams> DM::Nu()const{return ParamPolicy::prm();}

  uint DM::size()const{return nu().size();}
  const Vec & DM::nu() const{return Nu()->value();}
  const double & DM::nu(uint i)const{return nu()[i];}
  void DM::set_nu(const Vec &newnu){Nu()->set(newnu);}

  Vec DM::pi()const{
    Vec ans(nu());
    double nc=ans.sum();
    return ans/nc;
  }
  double DM::pi(uint i)const{return nu(i)/nu().sum();}


  double DM::pdf(dPtr dp, bool logscale) const{
    return pdf(DAT(dp)->value(),logscale);}

  double DM::pdf(const Data *dp, bool logscale) const{
    return pdf(DAT(dp)->value(),logscale);}

  double DM::pdf(const Vec &pi, bool logscale) const{
    return ddirichlet(pi, nu(), logscale);}


  double DirichletModel::Logp(const Vec &p, Vec &g, Mat &h, uint lev) const{
    // Because sum(p)=1, there are only p.size()-1 free elements in p.
    // The constraint is enforced by expressing the first element of p
    // as a function of the other variables.  The corresponding elements
    // in g and h are zeroed.

    const Vec &n(nu());
    double ans = ddirichlet(p, n, true);
    if(lev>0){
      int m = p.size();
      g[0]=0;
      for(int i = 0+1; i<m; ++i){
 	g[i]  = (n[i]-1)/p[i] - (n[0]-1)/p[0];
 	if(lev>1){
 	  h(0,i) = h(i,0)=0.0;
 	  for(int j = 0+1; j<m; ++j){
 	    h(i,j) = (1.0-n[0])/(p[0]*p[0])
 	      + (i==j ? (1.0-n[i])/(p[i]*p[i]) : 0.0);}}}}
    return ans;}


  //======================================================================
  double DirichletModel::Loglike(
      const Vector &nu, Vec & g, Mat &h, uint nd) const{

    /* returns log likelihood for the parameters of a Dirichlet
       distribution with sufficient statistic sumlogpi(lo..hi).  If
       pi(1)(lo..hi)..pi(nobs)(lo..hi) are probability vectors, then
       sumlogpi(j) = sum_i log(pi(i,j))

       if(nd>0) then the g(lo..hi) is filled with the gradient (with
       respect to nu).  If nd>1 then hess(lo..hi)(lo..hi) is filled
       with the hessian (wrt nu).  Otherwise the algorithm can be called
       with either g or hess = 0.

    */

    const Vec &sumlogpi(suf()->sumlog());
    double nobs = suf()->n();
    Vec *G(nd>0 ? &g : 0);
    Mat *H(nd>1 ? &h : 0);
    return dirichlet_loglike(nu, G, H, sumlogpi, nobs);
  }

  Vec DirichletModel::sim() const { return rdirichlet(nu()); }

  void DirichletModel::add_mixture_data(Ptr<Data> dp, double prob){
    suf()->add_mixture_data(DAT(dp)->value(), prob);
  }
}
