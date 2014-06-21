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

#ifndef BOOM_MULTINOMIAL_LOGIT_COMPLETE_DATA_SUF_HPP_
#define BOOM_MULTINOMIAL_LOGIT_COMPLETE_DATA_SUF_HPP_

#include <LinAlg/SpdMatrix.hpp>
#include <Models/Glm/ChoiceData.hpp>

namespace BOOM {
  class MultinomialLogitCompleteDataSufficientStatistics
      : private RefCounted{
  public:
    friend void intrusive_ptr_add_ref(
        MultinomialLogitCompleteDataSufficientStatistics *d){
      d->up_count();}
    friend void intrusive_ptr_release(
        MultinomialLogitCompleteDataSufficientStatistics *d){
      d->down_count();
      if(d->ref_count()==0) delete d;}

    MultinomialLogitCompleteDataSufficientStatistics(uint dim);
    MultinomialLogitCompleteDataSufficientStatistics * clone() const;

    void clear();
    void update(const Ptr<ChoiceData> dp, const Vector &wgts, const Vector &u);
    void add(Ptr<MultinomialLogitCompleteDataSufficientStatistics>);

    const Spd & xtwx()const;
    const Vector & xtwu()const;
    double weighted_sum_of_squares() const;

  private:
    mutable Spd xtwx_;
    Vector xtwu_;
    mutable bool sym_;
    double weighted_sum_of_squares_;
  };
}  // namespace BOOM

#endif // BOOM_MULTINOMIAL_LOGIT_COMPLETE_DATA_SUF_HPP_
