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
#ifndef BOOM_CONSTRAINED_VECTOR_PARAMS
#define BOOM_CONSTRAINED_VECTOR_PARAMS
#include <Models/ParamTypes.hpp>
namespace BOOM{

  class VectorConstraint : private RefCounted{
  public:
    friend void intrusive_ptr_add_ref(VectorConstraint *d){d->up_count();}
    friend void intrusive_ptr_release(VectorConstraint *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

    virtual ~VectorConstraint(){}

    virtual bool check(const Vec &v)const=0;
    // returns true if constraint satisfied

    virtual void impose(Vec &v)const=0;
    // forces constraint to hold

    virtual Vec expand(const Vec &small)const=0;
    // returns constrained vector from minimal information vector

    virtual Vec reduce(const Vec &large)const=0;
    // returns minimal information vector from constrained vector
  };
  //------------------------------------------------------------
  class NoConstraint : public VectorConstraint{
  public:
    bool check(const Vec &)const{return true;}
    void impose(Vec &)const{}
    Vec expand(const Vec &v)const{return v;}
    Vec reduce(const Vec &v)const{return v;}
  };
  //------------------------------------------------------------
  class ElementConstraint : public VectorConstraint{
  public:
    ElementConstraint(uint element=0, double x=0.0);
    virtual bool check(const Vec &v)const;
    virtual void impose(Vec &v)const;
    virtual Vec expand(const Vec &v)const;
    virtual Vec reduce(const Vec &v)const;
  private:
    uint element_;
    double value_;
  };
  //------------------------------------------------------------
  class SumConstraint : public VectorConstraint{
  public:
    SumConstraint(double sum);
    virtual bool check(const Vec &v)const;
    virtual void impose(Vec &v)const;
    virtual Vec expand(const Vec &v)const;  // adds final element to
    virtual Vec reduce(const Vec &v)const;  // eliminates last element
  private:
    double sum_;
  };

  //======================================================================

  class ConstrainedVectorParams : public VectorParams
  {
  public:
    explicit ConstrainedVectorParams(uint p, double x=0.0,
                                     Ptr<VectorConstraint> vc=0);
    ConstrainedVectorParams(const Vec &v,
                            Ptr<VectorConstraint> vc=0);  // copies v's data
    ConstrainedVectorParams(const ConstrainedVectorParams &rhs); // copies data
    ConstrainedVectorParams * clone()const;

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);

    bool check_constraint()const;
  private:
    Ptr<VectorConstraint> c_;
  };
  //------------------------------------------------------------


}
#endif// BOOM_CONSTRAINED_VECTOR_PARAMS
