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

#ifndef TARGET_FUN_H
#define TARGET_FUN_H

#include <LinAlg/Types.hpp>
#include <cpputil/ThrowException.hpp>
#include <cpputil/RefCounted.hpp>
#include <LinAlg/Vector.hpp>

namespace BOOM{
  // function object which can be passed to optimization routines

  class TargetFun : private RefCounted{
  public:

    virtual double operator()(const Vec &x)const=0;
    virtual ~TargetFun(){}
    friend void intrusive_ptr_add_ref(TargetFun *);
    friend void intrusive_ptr_release(TargetFun *);
  };
  void intrusive_ptr_add_ref(TargetFun *);
  void intrusive_ptr_release(TargetFun *);
  //----------------------------------------------------------------------
  class dTargetFun : virtual public TargetFun{
    double eps_scale;
  public:
    dTargetFun();
    virtual double operator()(const Vec &x)const=0;
    virtual double operator()(const Vec &x, Vec &g)const=0;
    void set_eps(double eps){eps_scale=eps;}
    //    Mat h_approx(const Vec &x)const;
  };
  //----------------------------------------------------------------------
  class d2TargetFun : virtual public dTargetFun{
  public:
    virtual double operator()(const Vec &x)const=0;
    virtual double operator()(const Vec &x, Vec &g)const=0;
    virtual double operator()(const Vec &x, Vec &g, Mat &h)const=0;
  };
  //======================================================================

  class ScalarTargetFun : private RefCounted{
  public:
    virtual double operator()(double x)const=0;
    virtual ~ScalarTargetFun(){}
    friend void intrusive_ptr_add_ref(ScalarTargetFun *);
    friend void intrusive_ptr_release(ScalarTargetFun *);
  };
  void intrusive_ptr_add_ref(ScalarTargetFun *);
  void intrusive_ptr_release(ScalarTargetFun *);
  //----------------------------------------------------------------------
  class dScalarTargetFun : virtual public ScalarTargetFun{
    double eps_scale;
  public:
    dScalarTargetFun();
    virtual double operator()(double x)const=0;
    virtual double operator()(double x, double &d)const=0;
    void eps(double e);
    double eps()const;
    //    double h_approx(const double &x)const;
  };
  //----------------------------------------------------------------------
  class d2ScalarTargetFun : virtual public dScalarTargetFun{
  public:
    virtual double operator()(double x)const=0;
    virtual double operator()(double x, double &d)const=0;
    virtual double operator()(double x, double &g, double &h)const=0;
  };

  //======================================================================

  class ScalarTargetView : public ScalarTargetFun{
  public:
    ScalarTargetView(TargetFun &F, const Vec &X, uint which_dim);
    virtual double operator()(double x)const;
    void set_x(const Vec &X);
  private:
    TargetFun &f;
    mutable Vec wsp;
    uint which;
  };

}
#endif // TARGET_FUN_H
