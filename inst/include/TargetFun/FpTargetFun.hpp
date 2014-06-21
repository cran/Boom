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

#ifndef FP_TARGET_FUN_H
#define FP_TARGET_FUN_H
#include <TargetFun/TargetFun.hpp>
#include <BOOM.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>

namespace BOOM{
  class FpTargetFun : virtual public TargetFun {
  public:
    typedef double (*fptr)(const Vec &v);
  public:
    FpTargetFun(fptr f) :fp(f){ }
    double operator()(const Vec &x)const{ return fp(x);}
  private:
    fptr fp;
  };


  class dFpTargetFun
    : virtual public dTargetFun{
  public:
    typedef double (*dfptr)(const Vec &x, Vec &g, bool dograd);
    dFpTargetFun(dfptr f) : fp(f){}
    double operator()(const Vec &x)const{ Vec g; return fp(x,g, false); }
    double operator()(const Vec &x, Vec &g)const{return  fp(x,g, true);}
  private:
    dfptr fp;
  };



  class d2FpTargetFun
    : virtual public d2TargetFun{
  public:
    typedef double (*d2fptr)(const Vec &x, Vec &g, Mat &h, uint nd);
    double operator()(const Vec &x)const{Vec g; Mat h; return fp(x,g,h,0);}
    double operator()(const Vec &x, Vec &g)const{Mat h; return fp(x,g,h,1);}
    double operator()(const Vec &x, Vec &g, Mat &h)const{ return fp(x,g,h,2);}
  private:
    d2fptr fp;
  };


  /*----------------------------------------------------------------------*/


}
#endif //FP_TARGET_FUN_H



