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

#ifndef BOOM_LOG_POST_H
#define BOOM_LOG_POST_H

#include <cpputil/Ptr.hpp>
#include <TargetFun/Loglike.hpp>
#include <numopt.hpp>

namespace BOOM{
  class VectorModel;
  class dVectorModel;
  class d2VectorModel;

  class LogPostTF{
  public:
    LogPostTF(Target Loglike, Ptr<VectorModel> Pri);
    double operator()(const Vector &z)const;
  private:
    Target loglike;
    Ptr<VectorModel> pri;
  };
  /*----------------------------------------------------------------------*/
  class dLogPostTF : public LogPostTF{
  public:
    dLogPostTF(dLoglikeTF Loglike, Ptr<dVectorModel>);
    dLogPostTF(Target Loglike, dTarget dLoglike, Ptr<dVectorModel>);
    double operator()(const Vector &z)const{
      return LogPostTF::operator()(z);}
    double operator()(const Vector &z, Vector &g)const;
  private:
    dTarget dloglike;
    Ptr<dVectorModel> dpri;
  };

  //----------------------------------------------------------------------
  class d2LogPostTF : public dLogPostTF{
  public:
    d2LogPostTF(d2LoglikeTF Loglike, Ptr<d2VectorModel> dp);
    d2LogPostTF(Target Loglike, dTarget dLoglike, d2Target d2Loglike,
		Ptr<d2VectorModel> dp);

    double operator()(const Vector &z)const{
      return LogPostTF::operator()(z);}
    double operator()(const Vector &z, Vector &g){
      return dLogPostTF::operator()(z,g);}
    double operator()(const Vector &z, Vector &g, Matrix &h)const;
  private:
    boost::function<double(const Vector &x, Vector &g, Mat&h)> d2loglike;
    Ptr<d2VectorModel> d2pri;
  };

}
#endif  // BOOM_LOG_POST_HPP
