/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#include <Models/Glm/PosteriorSamplers/MultinomialLogitCompleteDataSuf.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  namespace {
    typedef MultinomialLogitCompleteDataSufficientStatistics MLVSS;
  }

  MLVSS::MultinomialLogitCompleteDataSufficientStatistics(uint dim)
    : xtwx_(dim),
      xtwu_(dim),
      sym_(false),
      weighted_sum_of_squares_(0.0)
  {}

  MLVSS * MLVSS::clone() const {return new MLVSS(*this);}

  void MLVSS::clear(){
    xtwx_ = 0;
    xtwu_ = 0;
    weighted_sum_of_squares_ = 0.0;
    sym_ = false;
  }

  void MLVSS::update(Ptr<ChoiceData> dp, const Vec & wgts, const Vec &u){
    const Mat & X(dp->X(false));      // 'false' means omit columns
    xtwx_.add_inner(X, wgts, false);   // corresponding to subject X's at
    xtwu_ += X.Tmult(wgts*u);         // choice level 0.
    sym_ = false;
    for (int i = 0; i < wgts.size(); ++i) {
      weighted_sum_of_squares_ += wgts[i] * square(u[i]);
    }
  }

  void MLVSS::add(Ptr<MLVSS> s){
    xtwx_ += s->xtwx();
    xtwu_ += s->xtwu();
    sym_ = false;
    weighted_sum_of_squares_ += s->weighted_sum_of_squares();
  }

  const Spd & MLVSS::xtwx()const{
    if(!sym_) xtwx_.reflect();
    sym_ = true;
    return xtwx_;
  }

  const Vec & MLVSS::xtwu()const{return xtwu_;}

  double MLVSS::weighted_sum_of_squares() const {
    return weighted_sum_of_squares_;
  }
}
