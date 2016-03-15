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
    MH_Proposal(RNG &seeding_rng);
    ~MH_Proposal() override{}
    virtual Vector draw(const Vector &old)const=0;
    virtual double logf(const Vector &x, const Vector &old)const=0;
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
    MvtMhProposal(const SpdMatrix & Ivar, double nu,
                  RNG & seeding_rng = GlobalRng::rng);
    Vector draw(const Vector & old)const override;
    double logf(const Vector & x, const Vector & old)const override;
    virtual const Vector & mu(const Vector &old)const=0;
    void set_ivar(const SpdMatrix & Siginv);
    void set_var(const SpdMatrix & Sigma);
    void set_nu(double nu);
    uint dim()const;
    const SpdMatrix & ivar()const{return siginv_;}
   private:
    SpdMatrix siginv_;
    double ldsi_;
    Matrix chol_;  // lower or upper cholesky triangle (depending on
                // whether set_var or set_ivar is called)
                // Satisfies chol_ * chol_.t() == solve(siginv_)
    double nu_;
  };

  class MvtIndepProposal : public MvtMhProposal{
   public:
    MvtIndepProposal(const Vector & mu, const SpdMatrix & Ivar, double nu,
                     RNG & seeding_rng = GlobalRng::rng);
    bool sym()const override{return false;}
    const Vector & mu(const Vector &)const override{return mu_;}
    void set_mu(const Vector & mu);
    // the name 'mode' is used because 'mu' is taken
    const Vector & mode()const{return mu_;}
   private:
    Vector mu_;
  };

  class MvtRwmProposal : public MvtMhProposal{
   public:
    MvtRwmProposal(const SpdMatrix &Ivar, double nu,
                   RNG & seeding_rng = GlobalRng::rng);
    bool sym()const override{return true;}
    const Vector & mu(const Vector & old)const override{return old;}
   private:
  };

  class MvnIndepProposal : public MvtIndepProposal{
  public:
    MvnIndepProposal(const Vector & mu, const SpdMatrix &Ivar)
        : MvtIndepProposal(mu, Ivar, -1)
    {}
  };

  class MvnRwmProposal : public MvtRwmProposal{
  public:
    MvnRwmProposal(const SpdMatrix &Ivar)
        : MvtRwmProposal(Ivar, -1)
    {}
  };

  // ======================================================================
  // scalar proposals for Metropolis-Hastings algorithms
  class MH_ScalarProposal : private RefCounted{
  public:
    MH_ScalarProposal(RNG & seeding_rng = GlobalRng::rng);
    ~MH_ScalarProposal() override{}
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
    TScalarMhProposal(double sig, double nu,
                      RNG & seeding_rng = GlobalRng::rng);
    virtual double mu(double old)const=0;
    double draw(double old)const override;
    double logf(double x, double old)const override;
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
    TScalarRwmProposal(double sig, double nu,
                       RNG & seeding_rng = GlobalRng::rng)
        : TScalarMhProposal(sig, nu, seeding_rng)
    {}
    double mu(double old)const override{return old;}
    bool sym()const override{return true;}
  };
  // ----------------------------------------------------------------------
  class TScalarIndepProposal : public TScalarMhProposal{
   public:
    TScalarIndepProposal(double mu, double sigma, double nu,
                         RNG & seeding_rng = GlobalRng::rng)
        : TScalarMhProposal(sigma,nu, seeding_rng),
          mu_(mu)
    {}
    double mu(double)const override{return mu_;}
    bool sym()const override{return false;}
   private:
    double mu_;
  };
  // ----------------------------------------------------------------------
  class GaussianScalarRwmProposal : public TScalarRwmProposal{
   public:
    GaussianScalarRwmProposal(double sigma, RNG & seeding_rng = GlobalRng::rng)
        : TScalarRwmProposal(sigma, -1, seeding_rng)
    {}
  };
  // ----------------------------------------------------------------------
  class GaussianScalarIndepProposal : public TScalarIndepProposal{
   public:
    GaussianScalarIndepProposal(double mu, double sigma,
                                RNG & seeding_rng = GlobalRng::rng)
        : TScalarIndepProposal(mu, sigma, -1, seeding_rng)
    {}
  };

}
#endif// BOOM_MH_PROPOSALS_HPP
