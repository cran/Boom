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

#include <TargetFun/TargetFun.hpp>
#include <cmath>
#include <LinAlg/Matrix.hpp>
#include <cpputil/report_error.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
// #include <LinAlg/Types.hpp>

namespace BOOM{


  void intrusive_ptr_add_ref(TargetFun *s){
    s->up_count();}
  void intrusive_ptr_release(TargetFun *s){
    s->down_count();
    if(s->ref_count()==0) delete s; }

  dTargetFun::dTargetFun() : eps_scale(1e-5){}

  //======================================================================
  void intrusive_ptr_add_ref(ScalarTargetFun *s){
    s->up_count();}
  void intrusive_ptr_release(ScalarTargetFun *s){
    s->down_count();
    if(s->ref_count()==0) delete s; }
  //----------------------------------------------------------------------
  dScalarTargetFun::dScalarTargetFun() : eps_scale(1e-5){}


  d2TargetFunPointerAdapter::d2TargetFunPointerAdapter(
      const TargetType &target)
  {
    add_function(target);
  }

  d2TargetFunPointerAdapter::d2TargetFunPointerAdapter(
      const TargetType &prior, const TargetType &likelihood)
  {
    add_function(prior);
    add_function(likelihood);
  }

  void d2TargetFunPointerAdapter::add_function(const TargetType &fun) {
    targets_.push_back(fun);
  }

  double d2TargetFunPointerAdapter::operator()(const Vector &x) const {
    check_not_empty();
    double ans = targets_[0](x, nullptr, nullptr, true);
    for (int i = 1; i < targets_.size(); ++i) {
      ans += targets_[i](x, nullptr, nullptr, false);
    }
    return ans;
  }

  double d2TargetFunPointerAdapter::operator()(
      const Vector &x, Vector &g) const {
    check_not_empty();
    double ans = targets_[0](x, &g, nullptr, true);
    for (int i = 1; i < targets_.size(); ++i) {
      ans += targets_[i](x, &g, nullptr, false);
    }
    return ans;
  }

  double d2TargetFunPointerAdapter::operator()(
      const Vector &x, Vector &g, Matrix &h) const {
    check_not_empty();
    double ans = targets_[0](x, &g, &h, true);
    for (int i = 1; i < targets_.size(); ++i) {
      ans += targets_[i](x, &g, &h, false);
    }
    return ans;
  }

  void d2TargetFunPointerAdapter::check_not_empty() const {
    if (targets_.empty()) {
      report_error("Error in d2TargetFunPointerAdapter.  "
                   "No component functions specified.");
    }
  }

  //======================================================================

  ScalarTargetView::ScalarTargetView(TargetFun &F, const Vec &X, uint which_dim)
    : f(F),
      wsp(X),
      which(which_dim)
  {}

  double ScalarTargetView::operator()(double x)const{
    wsp[which] = x;
    return f(wsp);
  }

  void ScalarTargetView::set_x(const Vec &X){ wsp = X;}
}
