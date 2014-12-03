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
#include <LinAlg/Vector.hpp>
#include <boost/function.hpp>
#include <cpputil/RefCounted.hpp>
#include <cpputil/ThrowException.hpp>

namespace BOOM{
  // A suite function object which can be passed to optimization
  // routines.
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

  //----------------------------------------------------------------------
  // A common (and superior) pattern for writing functions and
  // derivatives is to pass the derivatives as pointers.  This class
  // is an adapter to convert that pattern to the one expected by
  // various function optimizers.
  //
  // This object evaluates to the sum of one or more functions with
  // the signature specified by TargetType.  The function arguments
  // are as follows.
  //   beta: The function argument.
  //   gradient: If non-NULL the gradient is computed and output
  //     here.  If NULL then no derivative computations are made.
  //   Hessian: If Hessian and gradient are both non-NULL the
  //     Hessian is computed and output here.  If NULL then the
  //     Hessian is not computed.
  //   reset_derivatives: If true then a non-NULL gradient or
  //     Hessian will be resized and set to zero.  If false then a
  //     non-NULL gradient or Hessian will have derivatives of
  //     log-liklihood added to its input value.  It is an error if
  //     reset_derivatives is false and the wrong-sized non-NULL
  //     argument is passed.
  class d2TargetFunPointerAdapter : public d2TargetFun {
   public:
    typedef boost::function<double(const Vector &x,
                                   Vector *gradient,
                                   Matrix *Hessian,
                                   bool reset_derivatives)> TargetType;
    d2TargetFunPointerAdapter() {}
    d2TargetFunPointerAdapter(const TargetType &target);
    d2TargetFunPointerAdapter(const TargetType &prior,
                              const TargetType &likelihood);
    void add_function(const TargetType &target);

    virtual double operator()(const Vector &x) const;
    virtual double operator()(const Vector &x, Vector &gradient) const;
    virtual double operator()(const Vector &x,
                              Vector &gradient,
                              Matrix &Hessian) const;
    // If targets_ is empty then an error is reported (e.g. by
    // throwing an exception).
    void check_not_empty() const;
   private:
    std::vector<TargetType> targets_;
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
