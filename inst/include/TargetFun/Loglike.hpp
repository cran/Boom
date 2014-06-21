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

#ifndef MODEL_TF_H
#define MODEL_TF_H

#include <Models/ModelTypes.hpp>

namespace BOOM{
  class ParamVecHolder;

  class LoglikeTF{
  public:
    LoglikeTF(LoglikeModel *);
    double operator()(const Vec &x)const;
    void swap_params(const Vec &x)const;   // puts x in model
    void restore_params()const;            // puts stored params back in model
  private:
    LoglikeModel * mod;   // provides loglike();
    mutable Vec wsp_;
  };
  //----------------------------------------------------------------------

  class dLoglikeTF : public LoglikeTF{
  public:
    dLoglikeTF(dLoglikeModel * d);
    double operator()(const Vec &x)const{return LoglikeTF::operator()(x);}
    double operator()(const Vec &x, Vec &g)const;
  private:
    dLoglikeModel * dmod;
  };

  //----------------------------------------------------------------------
  class d2LoglikeTF : public dLoglikeTF{
  public:
    d2LoglikeTF(d2LoglikeModel * d2);
    double operator()(const Vec &x)const{ return LoglikeTF::operator()(x);}
    double operator()(const Vec &x, Vec &g)const{
      return dLoglikeTF::operator()(x,g);}
    double operator()(const Vec &x, Vec &g, Mat &h)const;
  private:
    d2LoglikeModel * d2mod;
  };
  //------------------------------------------------------------
}
#endif // MODEL_TF_H


