/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_MH_PROPOSALS_HPP
#define BOOM_MH_PROPOSALS_HPP

#include <Samplers/Sampler.hpp>
#include <cpputil/Ptr.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>

namespace BOOM{
  // ======================================================================
  // MH_Proposal models a proposal distribution for a
  // MetropolisHastings sampler
  class MH_Proposal : private RefCounted{
  public:
    MH_Proposal();
    virtual ~MH_Proposal(){}
    virtual Vec draw(const Vec &old)const=0;
    virtual double logf(const Vec &x, const Vec &old)const=0;
    virtual bool sym()const=0;  // logf(x|old)== logf(old|x)

    friend void intrusive_ptr_add_ref(MH_Proposal *s){s->up_count();}
    friend void intrusive_ptr_release(MH_Proposal *s){
      s->down_count(); if(s->ref_count()==0) delete s;}

    RNG & rng()const{return rng_;}
   private:
    mutable RNG rng_;
  };
  // ======================================================================
  // Multivariate T proposal for Metropolis-Hastings samplers.  This
  // class is intended to be specialized for RWM and Independence
  // Metropolis, and to include Gaussian proposals by setting the
  // degrees of freedom parameter to either a negative number or to
  // infinity.
  class MvtMhProposal : public MH_Proposal{
   public:
    MvtMhProposal(const Spd & Ivar, double nu);
    virtual Vec draw(const Vec & old)const;
    virtual double logf(const Vec & x, const Vec & old)const;
    virtual const Vec & mu(const Vec &old)const=0;
    void set_ivar(const Spd & Siginv);
    void set_var(const Spd & Sigma);
    void set_nu(double nu);
    uint dim()const;
    const Spd & ivar()const{return siginv_;}
   private:
    Spd siginv_;
    double ldsi_;
    Mat chol_;  // lower or upper cholesky triangle (depending on
                // whether set_var or set_ivar is called)
                // Satisfies chol_ * chol_.t() == solve(siginv_)
    double nu_;
  };

  class MvtIndepProposal : public MvtMhProposal{
   public:
    MvtIndepProposal(const Vec & mu, const Spd & Ivar, double nu);
    virtual bool sym()const{return false;}
    virtual const Vec & mu(const Vec &)const{return mu_;}
    void set_mu(const Vec & mu);
    // the name 'mode' is used because 'mu' is taken
    const Vec & mode()const{return mu_;}
   private:
    Vec mu_;
  };

  class MvtRwmProposal : public MvtMhProposal{
   public:
    MvtRwmProposal(const Spd &Ivar, double nu);
    virtual bool sym()const{return true;}
    virtual const Vec & mu(const Vec & old)const{return old;}
   private:
  };

  class MvnIndepProposal : public MvtIndepProposal{
  public:
    MvnIndepProposal(const Vec & mu, const Spd &Ivar)
        : MvtIndepProposal(mu, Ivar, -1)
    {}
  };

  class MvnRwmProposal : public MvtRwmProposal{
  public:
    MvnRwmProposal(const Spd &Ivar)
        : MvtRwmProposal(Ivar, -1)
    {}
  };

  // ======================================================================
  // scalar proposals for Metropolis-Hastings algorithms
  class MH_ScalarProposal : private RefCounted{
  public:
    MH_ScalarProposal();
    virtual ~MH_ScalarProposal(){}
    virtual double draw(double old)const=0;
    virtual double logf(double x, double old)const=0;
    virtual bool sym()const=0;

    friend void intrusive_ptr_add_ref(MH_ScalarProposal *s){s->up_count();}
    friend void intrusive_ptr_release(MH_ScalarProposal *s){
      s->down_count(); if(s->ref_count()==0) delete s;}
    RNG & rng()const{return rng_;}
   private:
    mutable RNG rng_;
  };
  // ----------------------------------------------------------------------
  class TScalarMhProposal : public MH_ScalarProposal{
   public:
    TScalarMhProposal(double sig, double nu);
    virtual double mu(double old)const=0;
    virtual double draw(double old)const;
    virtual double logf(double x, double old)const;
    void set_sigma(double sig){sig_ = sig;}
    void set_nu(double nu){nu_ = nu;}
    double sigma()const{return sig_;}
    double nu()const{return nu_;}
   private:
    double sig_;
    double nu_;
  };
  // ----------------------------------------------------------------------
  class TScalarRwmProposal : public TScalarMhProposal{
   public:
    TScalarRwmProposal(double sig, double nu)
        : TScalarMhProposal(sig, nu)
    {}
    virtual double mu(double old)const{return old;}
    virtual bool sym()const{return true;}
  };
  // ----------------------------------------------------------------------
  class TScalarIndepProposal : public TScalarMhProposal{
   public:
    TScalarIndepProposal(double mu, double sigma, double nu)
        : TScalarMhProposal(sigma,nu),
          mu_(mu)
    {}
    virtual double mu(double)const{return mu_;}
    virtual bool sym()const{return false;}
   private:
    double mu_;
  };
  // ----------------------------------------------------------------------
  class GaussianScalarRwmProposal : public TScalarRwmProposal{
   public:
    GaussianScalarRwmProposal(double sigma)
        : TScalarRwmProposal(sigma, -1)
    {}
  };
  // ----------------------------------------------------------------------
  class GaussianScalarIndepProposal : public TScalarIndepProposal{
   public:
    GaussianScalarIndepProposal(double mu, double sigma)
        : TScalarIndepProposal(mu, sigma, -1)
    {}
  };

}
#endif// BOOM_MH_PROPOSALS_HPP
