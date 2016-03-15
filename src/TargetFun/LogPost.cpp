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

#include <TargetFun/LogPost.hpp>
#include <Models/VectorModel.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{



  LogPostTF::LogPostTF(Target m, Ptr<VectorModel> p)
    : loglike(m),
      pri(p)
  {}

  //--------------------------------------------------
  dLogPostTF::dLogPostTF(dLoglikeTF m, Ptr<dVectorModel> p)
    : LogPostTF(m,p),
      dloglike(m),
      dpri(p)
  {}

  dLogPostTF::dLogPostTF(Target m, dTarget dm, Ptr<dVectorModel> p)
    : LogPostTF(m,p),
      dloglike(dm),
      dpri(p)
  {}


  //--------------------------------------------------
  d2LogPostTF::d2LogPostTF(d2LoglikeTF m, Ptr<d2VectorModel> p)
    : dLogPostTF(m,p),
      d2loglike(m),
      d2pri(p)
  {}


  d2LogPostTF::d2LogPostTF(Target m, dTarget dm, d2Target d2m,
			   Ptr<d2VectorModel> p)
    : dLogPostTF(m, dm,p),
      d2loglike(d2m),
      d2pri(p)
  {}

  //--------------------------------------------------

  double LogPostTF::operator()(const Vector &z)const{
    double ans = pri->logp(z);
    if(ans==BOOM::negative_infinity()) return ans;
    ans += loglike(z);
    return ans;
  }

  //----------------------------------------------------------------------

  double dLogPostTF::operator()(const Vector &x, Vector &g)const{
    g=0.0;
    Vector g1 = g;
    double ans = dloglike(x,g);
    ans += dpri->dlogp(x, g1);
    g+=g1;
    return ans;
  }

  //----------------------------------------------------------------------
  double d2LogPostTF::operator()(const Vector &x, Vector &g, Matrix &h)const{
    g=0.0;
    Vector g1 = g;
    h=0.0;
    Matrix h1 = h;
    double ans = d2loglike(x, g, h);  // derivatives wrt x
    ans += d2pri->d2logp(x, g1, h1);       // derivatives wrt x
    g+=g1;
    h+=h1;
    return ans;
  }

}
