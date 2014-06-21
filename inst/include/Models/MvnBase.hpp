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

#ifndef BOOM_MVN_MODEL_BASE_HPP
#define BOOM_MVN_MODEL_BASE_HPP
#include <LinAlg/Selector.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/VectorModel.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Sufstat.hpp>
#include <Models/DataTypes.hpp>
#include <Models/SpdParams.hpp>

namespace BOOM{

   class MvnSuf: public SufstatDetails<VectorData>{
    public:
     // If created using the default constructor, the MvnSuf will be
     // resized to the dimension of the first data point passed to it
     // in update().
     MvnSuf(uint p=0);
     MvnSuf(double n, const Vec &ybar, const Spd &sumsq);
     MvnSuf(const MvnSuf &sf);
     MvnSuf *clone() const;

     void clear();
     void resize(uint p);  // clears existing data
     void Update(const VectorData &x);
     void update_raw(const Vec &x);
     void add_mixture_data(const Vec &x, double prob);

     Vec sum()const;
     Spd sumsq()const;       // Un-centered sum of squares
     double n()const;
     const Vec & ybar()const;
     Spd sample_var()const;  // divides by n-1
     Spd var_hat()const;     // divides by n
     Spd center_sumsq(const Vec &mu)const;
     const Spd & center_sumsq()const;

     void combine(Ptr<MvnSuf>);
     void combine(const MvnSuf &);
     MvnSuf * abstract_combine(Sufstat *s);

     virtual Vec vectorize(bool minimal=true)const;
     virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                             bool minimal=true);
     virtual Vec::const_iterator unvectorize(const Vec &v,
                                             bool minimal=true);

     virtual ostream & print(ostream &)const;
    private:
     Vec ybar_;
     Vec wsp_;
     mutable Spd sumsq_;     // centered at ybar
     double n_;              // sample size
     mutable bool sym_;
     void check_symmetry()const;

     // resizes if empty, otherwise throws if dimension is wrong.
     void check_dimension(const Vec &y);
   };

  inline ostream & operator<<(ostream &out, const MvnSuf &s){
    return s.print(out);}
  //------------------------------------------------------------

  class MvnBase
    : public DiffVectorModel
  {
  public:
    virtual MvnBase * clone()const=0;
    virtual uint dim()const;
    virtual double Logp(const Vec &x, Vec &g, Mat &h, uint nderiv)const;

    // Args:
    //   x_subset: A subset (determined by 'inclusion') of the vector
    //     of random variables measured by this model.
    //   gradient: If nderiv > 0 then gradient will be filled with the
    //     gradient of this function with respect to the dimensions of
    //     x determined by 'inclusion.'  Otherwise 'gradient' is not
    //     used.
    //   Hessian: If nderiv > 1 then Hessian will be filled with the
    //     matrix of second derivatives with respect to the dimensions
    //     of x determined by 'inclusion.'  Otherwise 'Hessian' is not
    //     used.
    //   nderiv:  The number of derivatives to take.
    //   inclusion:  The 'included' positions.
    //
    // Returns:
    //   The log of the normal density with mean mu[inclusion] and
    //   precision siginv[inclusion] evalueated at x_subset.
    virtual double logp_given_inclusion(const Vector &x_subset,
                                        Vector &gradient,
                                        Matrix &Hessian,
                                        int nderiv,
                                        const Selector &inclusion)const;
    virtual const Vec & mu() const=0;
    virtual const Spd & Sigma()const=0;
    virtual const Spd & siginv() const=0;
    virtual double ldsi()const=0;
    virtual Vec sim()const;
  };

  //____________________________________________________________
  class MvnBaseWithParams
    : public MvnBase,
      public ParamPolicy_2<VectorParams, SpdParams>,
      public LocationScaleVectorModel
  {
  public:
    MvnBaseWithParams(uint p, double mu=0.0, double sig=1.0);
    // N(mu,V)... if(ivar) then V is the inverse variance.
    MvnBaseWithParams(const Vec &mean, const Spd &V,
		      bool ivar=false);
    MvnBaseWithParams(Ptr<VectorParams>, Ptr<SpdParams>);
    MvnBaseWithParams(const MvnBaseWithParams &);
    MvnBaseWithParams * clone()const=0;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm()const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm()const;

    virtual const Vec & mu() const;
    virtual  const Spd & Sigma()const;
    virtual  const Spd & siginv() const;
    virtual double ldsi()const;

    virtual void set_mu(const Vec &);
    virtual void set_Sigma(const Spd &);
    virtual void set_siginv(const Spd &);
    virtual void set_S_Rchol(const Vec &sd, const Mat &L);
  };


}

#endif// BOOM_MVN_MODEL_BASE_HPP
