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

#ifndef BOOM_LOGLIKE_SUBSET_HPP
#define BOOM_LOGLIKE_SUBSET_HPP

#include <Models/ModelTypes.hpp>
#include <LinAlg/Selector.hpp>
#include <boost/shared_ptr.hpp>

namespace BOOM{
  class ParamVecHolder;
  class Selector;

  class LoglikeSubsetTF{
  public:
    // the following constructors can be used to obtain
    // (1) full log likelihood for all parameters
    // (2) conditional log likelihood for a subset
    // (3) conditional log likelihood for a single parameter

    LoglikeSubsetTF(LoglikeModel *);
    LoglikeSubsetTF(LoglikeModel *, const ParamVec &v);
    LoglikeSubsetTF(LoglikeModel *m, Ptr<Params> T);

    double operator()(const Vec &x)const;
  private:
    LoglikeModel *mod;   // provides loglike();
    mutable Vec wsp;
  protected:
    ParamVecHolder hold_params(const Vec &x)const;
    ParamVec t;      // params owned by mod;
  };
  //----------------------------------------------------------------------

  class Selector;
  class dLoglikeSubsetTF : public LoglikeSubsetTF{
  public:
    dLoglikeSubsetTF(dLoglikeModel *d);
    dLoglikeSubsetTF(dLoglikeModel *d, const ParamVec &);
    dLoglikeSubsetTF(dLoglikeModel *d, Ptr<Params> t);
    double operator()(const Vec &x)const{
      return LoglikeSubsetTF::operator()(x);}
    double operator()(const Vec &x, Vec &g)const;
  private:
    dLoglikeModel *dmod;
    boost::shared_ptr<Selector> pos_;
    void get_pos();  // called during construction
  protected:
    const Selector & pos()const;
  };

  //----------------------------------------------------------------------
  class d2LoglikeSubsetTF : public dLoglikeSubsetTF{
  public:
    d2LoglikeSubsetTF(d2LoglikeModel *d2);
    d2LoglikeSubsetTF(d2LoglikeModel *d2, Ptr<Params>);
    d2LoglikeSubsetTF(d2LoglikeModel *d2, const ParamVec &);

    double operator()(const Vec &x)const{
      return LoglikeSubsetTF::operator()(x);}
    double operator()(const Vec &x, Vec &g)const{
      return dLoglikeSubsetTF::operator()(x,g);}
    double operator()(const Vec &x, Vec &g, Mat &h)const;
  private:
    d2LoglikeModel *d2mod;
  };
  //------------------------------------------------------------
}
#endif // BOOM_LOGLIKE_SUBSET_HPP
