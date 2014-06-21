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

#include <Models/Glm/PosteriorSamplers/GlmMvnSiginvSepStratSampler.hpp>
#include <Samplers/ScalarSliceSampler.hpp>
#include <LinAlg/VectorView.hpp>
#include <LinAlg/Givens.hpp>
#include <cmath>
#include <LinAlg/Matrix.hpp>
#include <TargetFun/TargetFun.hpp>
#include <distributions.hpp>
#include <cpputil/ask_to_continue.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  typedef GlmMvnSiginvSepStratSampler GMSSS;

  class LTF : public ScalarTargetFun{
  public:
    typedef std::vector<Ptr<GlmCoefs> > DVEC;
    typedef std::vector<Ptr<GammaModel> > PVEC;
    LTF(uint i, uint j,
	GlmMvnPrior *mod,
	const DVEC &d,
	const PVEC &Spri,
	const Mat & sumsq_chol,
	Mat & LT,
	Mat & WSP,
	Spd & SigmaWsp,
	Vec & SWsp);

    double operator()(double ell)const;
    //    LTF * clone()const;
    uint dim()const;
  private:
    double eval_logdet()const;
    double eval_qform()const;
    double eval_prior()const;
    void compute_S()const;

    uint i,j;
    GlmMvnPrior *mod_;
    const DVEC & dat;         // used for eval_logdet
    const PVEC & Spri;
    const Mat & sumsq_chol;
    Mat & LT;
    Mat & Wsp;
    Spd & Sigma;
    Vec & S;
  };

  GMSSS::GlmMvnSiginvSepStratSampler(GlmMvnPrior *Mod,
				     std::vector<Ptr<GammaModel> > S_pri)
    : mod_(Mod),
      Spri(S_pri),
      LT(mod_->dim(), mod_->dim()),
      Wsp(LT),
      sumsq_chol(LT),
      nobs(mod_->dim()),
      Sigma(mod_->dim()),
      S(mod_->dim())
  {}


  void GMSSS::observe_sigma(const Spd &){
    L_current = false;
  }
  typedef std::vector<Ptr<GlmCoefs> > DVEC;

  void GMSSS::draw(){
    uint d = dim();
    Ptr<GlmMvnSuf> s(mod_->suf());
    sumsq_chol = s->center_sumsq(mod_->mu()).chol();
    LT = mod_ -> siginv_chol().t();
    nobs  = s->vnobs();

    for(uint i=0; i<d; ++i){
      for(uint j=0; j<=i; ++j){
	//	cout << "    i = " << i << "  j = " << j << endl;
	draw_L(i,j);}}
    set_Sigma();
  }

  void GMSSS::draw_L(uint i, uint j){
    const DVEC & d(mod_->dat());
    LTF logp(i,j, mod_,d, Spri, sumsq_chol, LT, Wsp, Sigma, S);
    double x = LT(j,i);
    ScalarSliceSampler sam(logp);
    if(i==j) sam.set_lower_limit(0);
    LT(j,i) = sam.draw(x);
  }

  void GMSSS::set_Sigma(){
    Sigma = chol2inv(LT.t());
    mod_->set_Sigma(Sigma);
  }

  double GMSSS::logpri()const{
    const Spd  & Sigma(mod_->Sigma());
    ConstVectorView v(diag(Sigma));
    const Corr R(var2cor(Sigma));
    uint n = S.size();
    double ans=0;
    Mat L(R.chol());
    Spd Rinv = chol2inv(L);
    for(uint i=0; i<n; ++i){
      ans += dgamma(1.0/v[i], Spri[i]->alpha(), Spri[i]->beta());
      ans -= 2*(n+1)*fabs(L(i,i));
      ans -= .5*(n+1)*Rinv(i,i);
    }
    return ans;
  }

  uint GMSSS::dim()const{ return Spri.size(); }
  //____________________________________________________________


  LTF::LTF(uint I, uint J,
	   GlmMvnPrior *mod,
	   const DVEC & D,
	   const PVEC &SPRI,
	   const Mat & SumsqChol,
	   Mat & Lt,
	   Mat & WSP,
	   Spd & SigmaWsp,
	   Vec & SWsp)
    : i(I),
      j(J),
      mod_(mod),
      dat(D),
      Spri(SPRI),
      sumsq_chol(SumsqChol),
      LT(Lt),
      Wsp(WSP),
      Sigma(SigmaWsp),
      S(SWsp)
  {}


  //  LTF * LTF::clone()const{return new LTF(*this);}

  double LTF::operator()(double ell)const{
    double old_ell = LT(j,i);
    LT(j,i) = ell;
    compute_S();
    double logdet = eval_logdet();
    double qform = eval_qform();
    double prior = eval_prior();
    LT(j,i) = old_ell;
    double ans = logdet + qform + prior;
    if(std::isnan(ans)){
      ostringstream err;
      err << "LTF produced a nan:  "
	  << " logdet = " << logdet
	  << "  qform = " << qform
	  << "  prior = " << prior << endl;
      throw_exception<std::runtime_error>(err.str());
    }
    return ans;
  }


  void LTF::compute_S()const{
    Sigma = chol2inv(LT.t());
    S = sqrt(diag(Sigma));
  }

  //------------------------------------------------------------

  double LTF::eval_logdet()const{
    uint J = dat.size();
    double ans = 0;
    for(uint j = 0; j<J; ++j){
      const Selector & inc(dat[j]->inc());
      Wsp = triangulate(LT,inc,false);
      const VectorView ell(Wsp.diag());
      uint n = ell.size();
      for(uint k=0; k<n; ++k) ans+= log(fabs(ell[k]));
    }
    return ans;
  }

  double LTF::eval_qform()const{
    Wsp = Umult(LT,sumsq_chol);
    double ans = -0.5 * traceAB(Wsp,Wsp.t());
    return ans;
  }
  //------------------------------------------------------------

  double LTF::eval_prior()const{
    double ans=0;
    uint K = dim();

    const VectorView ell_diag(LT.diag());
    for(uint k=0; k<K; ++k){
      if(S[k]==0) return BOOM::negative_infinity();
      double ss = 2*Spri[k]->beta();
      double df = 2*Spri[k]->alpha();
      ans -= .5* ss/pow(S[k],2);
      ans -= df * log(S[k]);

      // jacobian bit
      VectorView ell_full(LT.col(k));
      VectorView ell(ell_full, 0, k+1);
      double diagSiginv = ell.normsq();
      if(diagSiginv==0) return BOOM::negative_infinity();
      ans -= 0.5*(K+1) * log(fabs(diagSiginv));

      if(ell_diag[k]==0.0) return BOOM::negative_infinity();
      ans += (K-k) * log(fabs(ell_diag[k]));
    }
    return ans;
  }
  //------------------------------------------------------------
  uint LTF::dim()const{ return LT.nrow(); }
}
