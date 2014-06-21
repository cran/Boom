/*
  Copyright (C) 2007-2010 Steven L. Scott

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

#include <Models/MvnBase.hpp>
#include <distributions.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  typedef MvnBase MB;

  MvnSuf::MvnSuf(uint p)
    : ybar_(p, 0.0),
      sumsq_(p, 0.0),
      n_(0.0),
      sym_(false)
  {}

  MvnSuf::MvnSuf(double n, const Vec &ybar, const Spd & sumsq)
      : ybar_(ybar),
        sumsq_(sumsq),
        n_(n),
        sym_(false)
  {}

  MvnSuf::MvnSuf(const MvnSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<VectorData>(rhs),
      ybar_(rhs.ybar_),
      sumsq_(rhs.sumsq_),
      n_(rhs.n_),
      sym_(rhs.sym_)
  {}

  MvnSuf *MvnSuf::clone() const{ return new MvnSuf(*this);}

  void MvnSuf::clear(){
    ybar_=0;
    sumsq_=0;
    n_=0;
    sym_ = false;
  }

  void MvnSuf::resize(uint p){
    ybar_.resize(p);
    sumsq_.resize(p);
    clear();
  }

  void MvnSuf::check_dimension(const Vec &y){
    if(ybar_.size() == 0){
      resize(y.size());
    }
    if(y.size() != ybar_.size()){
      ostringstream msg;
      msg << "attempting to update MvnSuf of dimension << " << ybar_.size()
          << " with data of dimension " << y.size() << "." << endl
          << "Value of data point is [" << y << "]";
      throw_exception<std::runtime_error>(msg.str().c_str());
    }
  }

  void MvnSuf::update_raw(const Vec & y){
    check_dimension(y);
    n_+=1.0;
    wsp_ = (y - ybar_)/n_;  // old ybar, new n
    ybar_ += wsp_;          // new ybar
    sumsq_.add_outer(wsp_, n_-1, false);
    sumsq_.add_outer(y - ybar_, 1, false);
    sym_ = false;
  }

  void MvnSuf::Update(const VectorData &X){
    const Vec &x(X.value());
    update_raw(x);
  }

  void MvnSuf::add_mixture_data(const Vec &y, double prob){
    check_dimension(y);
    n_ += prob;
    wsp_ = (y - ybar_)*(prob/n_);  // old ybar_, new n_
    ybar_ += wsp_;                 // new ybar_
    sumsq_.add_outer(wsp_, n_ - prob, false);
    sumsq_.add_outer(y - ybar_, prob, false);
    sym_ = false;
  }

  Vec MvnSuf::sum()const{return ybar_ * n_;}
  Spd MvnSuf::sumsq()const{
    check_symmetry();
    Spd ans(sumsq_);
    ans.add_outer(ybar_, n_);
    return ans;
  }
  double MvnSuf::n()const{return n_;}

  void MvnSuf::check_symmetry()const{
    if(!sym_){
      sumsq_.reflect();
      sym_ = true;
    }
  }

  const Vec & MvnSuf::ybar()const{ return ybar_;}
  Spd MvnSuf::sample_var()const{
    if(n()>1) return center_sumsq()/(n()-1);
    return sumsq_ * 0.0;
  }

  Spd MvnSuf::var_hat()const{
    if(n()>0) return center_sumsq()/n();
    return sumsq_ * 0.0;
  }

  Spd MvnSuf::center_sumsq(const Vec &mu)const{
    Spd ans = center_sumsq();
    ans.add_outer(ybar_ - mu, n_);
    return ans;
  }

  const Spd & MvnSuf::center_sumsq()const{
    check_symmetry();
    return sumsq_;
  }

  void MvnSuf::combine(Ptr<MvnSuf> s){
    this->combine(*s);
  }

  // TODO(stevescott): test this
  void MvnSuf::combine(const MvnSuf & s){
    Vec zbar = (sum() + s.sum())/(n() + s.n());
    sumsq_ = center_sumsq(zbar) + s.center_sumsq(zbar);
    ybar_ = zbar;
    n_ += s.n();
    sym_ = true;
  }

  MvnSuf * MvnSuf::abstract_combine(Sufstat *s){
      return abstract_combine_impl(this,s); }

  Vec MvnSuf::vectorize(bool minimal)const{
    Vec ans(ybar_);
    ans.concat(sumsq_.vectorize(minimal));
    ans.push_back(n_);
    return ans;
  }

  Vec::const_iterator MvnSuf::unvectorize(Vec::const_iterator &v, bool){
    uint dim = ybar_.size();
    ybar_.assign(v, v+dim);
    v+=dim;
    sumsq_.unvectorize(v);
    n_ = *v; ++v;
    return v;
  }

  Vec::const_iterator MvnSuf::unvectorize(const Vec &v,
                                          bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & MvnSuf::print(ostream &out)const{
    out << n_ << endl
        << ybar_ << endl
        << sumsq_;
    return out;
  }

  //======================================================================

  uint MB::dim()const{
    return mu().size();}

  double MB::Logp(const Vec &x, Vec &g, Mat &h, uint nd)const{
    double ans = dmvn(x,mu(), siginv(), ldsi(), true);
    if(nd>0){
      g = -(siginv() * (x-mu()));
      if(nd>1) h = -siginv();}
    return ans;}

  double MB::logp_given_inclusion(const Vector &x_subset,
                                  Vector &gradient,
                                  Matrix &Hessian,
                                  int nd,
                                  const Selector &inc) const {
    Vector mu0 = inc.select(mu());
    SpdMatrix precision = inc.select(siginv());
    double ans = dmvn(x_subset, mu0, precision, precision.logdet(), true);
    if (nd > 0) {
      gradient = -precision * (x_subset - mu0);
      if (nd > 1) {
        Hessian = -precision;
      }
    }
    return ans;
  }

  Vec MB::sim()const{
    return rmvn(mu(), Sigma());
  }

  typedef MvnBaseWithParams MBP;

  MBP::MvnBaseWithParams(uint p, double mu, double sigsq)
    : ParamPolicy(new VectorParams(p,mu),
		  new SpdParams(p,sigsq))
  {}

    // N(mu,V)... if(ivar) then V is the inverse variance.
  MBP::MvnBaseWithParams(const Vec &mean, const Spd &V, bool ivar)
    : ParamPolicy(new VectorParams(mean), new SpdParams(V,ivar))
  {}


  MBP::MvnBaseWithParams(Ptr<VectorParams> mu, Ptr<SpdParams> S)
    : ParamPolicy(mu,S)
  {}

  MBP::MvnBaseWithParams(const MvnBaseWithParams &rhs)
    : Model(rhs),
      VectorModel(rhs),
      MvnBase(rhs),
      ParamPolicy(rhs),
      LocationScaleVectorModel(rhs)
  {}

  Ptr<VectorParams> MBP::Mu_prm(){
    return ParamPolicy::prm1();}
  const Ptr<VectorParams> MBP::Mu_prm()const{
    return ParamPolicy::prm1();}

  Ptr<SpdParams> MBP::Sigma_prm(){
    return ParamPolicy::prm2();}
  const Ptr<SpdParams> MBP::Sigma_prm()const{
    return ParamPolicy::prm2();}

  const Vec & MBP::mu()const{return Mu_prm()->value();}
  const Spd & MBP::Sigma()const{return Sigma_prm()->var();}
  const Spd & MBP::siginv()const{return Sigma_prm()->ivar();}
  double MBP::ldsi()const{return Sigma_prm()->ldsi();}

  void MBP::set_mu(const Vec &v){Mu_prm()->set(v);}
  void MBP::set_Sigma(const Spd &s){Sigma_prm()->set_var(s);}
  void MBP::set_siginv(const Spd &ivar){Sigma_prm()->set_ivar(ivar);}
  void MBP::set_S_Rchol(const Vec &sd, const Mat &L){
      Sigma_prm()->set_S_Rchol(sd,L); }


}
