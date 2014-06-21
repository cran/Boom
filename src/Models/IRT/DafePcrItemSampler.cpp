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
#include <Models/IRT/DafePcr.hpp>
#include <distributions.hpp>
#include <cpputil/ParamHolder.hpp>
#include <cpputil/math_utils.hpp>
#include <Models/MvnModel.hpp>
#include <Models/MvtModel.hpp>
#include <Models/IRT/PartialCreditModel.hpp>
#include <Models/IRT/Subject.hpp>
#include <Samplers/MetropolisHastings.hpp>
#include <TargetFun/TargetFun.hpp>
#include <boost/bind.hpp>
#include <iomanip>

namespace BOOM{
  namespace IRT{
    typedef DafePcrItemSampler ISAM;
    typedef PartialCreditModel PCR;
    typedef DafePcrDataImputer IMP;
    //======================================================================
    class PcrBetaHolder{
    public:
      PcrBetaHolder(const Vec &newb, Ptr<PartialCreditModel> pcr, Vec &V)
	: v(V), mod(pcr){ v = mod->beta(); mod->set_beta(newb); }
      ~PcrBetaHolder(){ mod->set_beta(v); }
    private:
      Vec &v;
      Ptr<PartialCreditModel> mod;
    };
    //======================================================================
    class ItemDafeTF : public TargetFun{
      // evaluates posterior probability of a vector b
    public:
      ItemDafeTF(Ptr<PCR> it, Ptr<MvnModel> pri, Ptr<IMP> Imp)
	: mod(it), prior(pri), imp(Imp), t(it->t()) {}
      double operator()(const Vec &b)const;
      ItemDafeTF * clone()const{return new ItemDafeTF(*this);}
    private:
      Ptr<PCR> mod;
      Ptr<MvnModel> prior;
      Ptr<IMP> imp;  // must be assigned to  'item'.
                     // contains imputed latent data
      mutable Vec tmpbeta;
      mutable double ans;
      ParamVec t;
      void logp_sub(Ptr<Subject> s)const;
    };
    void ItemDafeTF::logp_sub(Ptr<Subject> s)const{
      Response r = s->response(mod);
      const Vec & u(imp->get_u(r, true));
      const Vec & Theta(s->Theta());
      const Vec &eta(mod->fill_eta(Theta));
      assert(u.size()==eta.size());
      for(uint i=0; i<u.size(); ++i) ans+= dexv(u[i], eta[i], 1.0, true);
    }
    double ItemDafeTF::operator()(const Vec &b)const{
      PcrBetaHolder ph(b, mod, tmpbeta);
      if( mod->a() <=0) return BOOM::negative_infinity();
      const SubjectSet & subjects(mod->subjects());
      ans=0.0;
      for_each(subjects.begin(), subjects.end(),
	       boost::bind(&ItemDafeTF::logp_sub, this, _1));
      return ans;
    }
    //======================================================================
    ISAM::DafePcrItemSampler(Ptr<PCR> Mod, Ptr<DafePcrDataImputer> Imp,
			     Ptr<MvnModel> Prior, double Tdf )
      : mod(Mod),
	prior(Prior),
	imp(Imp),
	sigsq(1.644934066848226) // pi^2/6
    {
      Mat X(Mod->X(1.0));
      xtx = Spd(X.ncol());
      xtu = Vec(X.ncol());

      ItemDafeTF target(mod, prior, imp);
      uint dim = mod->beta().size();
      Spd Ominv(dim);
      Ominv.set_diag(1.0);
      prop = new MvtIndepProposal(Vec(dim), Ominv, Tdf);
      sampler = new MetropolisHastings(target, prop);
    }
    //------------------------------------------------------------
    double ISAM::logpri()const{ return prior->logp(mod->beta()); }
    //------------------------------------------------------------
    void ISAM::draw(){
      get_moments();    // fills xtx and xtu
      prop->set_mu(mean);
      prop->set_ivar(ivar);
      Vec b = mod->beta();
      b = sampler->draw(b);
      mod->set_beta(b);
      mod->sync_params();
    }
    //----------------------------------------------------------------------
    void ISAM::get_moments(){
      xtx=0.0;
      xtu = 0.0;
      const SubjectSet & s(mod->subjects());
      for_each(s.begin(), s.end(),
	       boost::bind(&ISAM::accumulate_moments, this, _1));
      ivar= as_symmetric(xtx)/sigsq+ prior->siginv();
      mean = ivar.solve(prior->siginv()*prior->mu() + xtu/sigsq);
    }
    //----------------------------------------------------------------------
    void ISAM::accumulate_moments(Ptr<Subject> s){
      const Mat &X(mod->X(s->Theta()));
      xtx.add_inner(X);
      Response r = s->response(mod);
      const Vec &u(imp->get_u(r, true));
      xtu.add_Xty(X, u);
    }
  }// namespace IRT
} // namespace BOOM
