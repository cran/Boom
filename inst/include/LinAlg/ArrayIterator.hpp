/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_ARRAY_ITERATOR_HPP_
#define BOOM_ARRAY_ITERATOR_HPP_

#include <vector>
#include <iterator>
#include <iostream>

namespace BOOM{

using std::cout;
using std::endl;

  class ArrayBase;
  class ConstArrayBase;

  class ArrayIterator
      : public std::iterator<std::forward_iterator_tag, double>
  {
   public:
    ArrayIterator(ArrayBase *host,
                  const std::vector<int> &starting_position);
    ArrayIterator(ArrayBase *host);

    double & operator*()const;

    bool operator==(const ArrayIterator &rhs)const{
      return (host_ == rhs.host_
              && dims_ == rhs.dims_
              && pos_ == rhs.pos_);}

    bool operator!=(const ArrayIterator &rhs)const{
      return !(*this == rhs); }

    ArrayIterator & operator++(){
      for(int d = 0; d < dims_.size(); ++d){
        ++pos_[d];
        if(pos_[d] < dims_[d]) return *this;
        pos_[d] = 0;
      }
      pos_ = dims_;
      for(int i = 1; i < dims_.size(); ++i) --pos_[i];
      return *this;
    }

    const std::vector<int> & pos()const{return pos_;}

   private:
    ArrayBase * host_;
    const std::vector<int> &dims_;
    std::vector<int> pos_;
  };

  //======================================================================
  class ConstArrayIterator
      : public std::iterator<std::forward_iterator_tag, double>
  {
   public:
    ConstArrayIterator(const ConstArrayBase *host,
                  const std::vector<int> &starting_position);
    ConstArrayIterator(const ConstArrayBase *host);

    double operator*()const;

    bool operator==(const ConstArrayIterator &rhs)const{
      return (host_ == rhs.host_
              && dims_ == rhs.dims_
              && pos_ == rhs.pos_);}

    bool operator!=(const ConstArrayIterator &rhs)const{
      return !(*this == rhs); }

    ConstArrayIterator & operator++(){
      for(int d = 0; d < dims_.size(); ++d){
        ++pos_[d];
        if(pos_[d] < dims_[d]) return *this;
        pos_[d] = 0;
      }
      return *this;
    }

    const std::vector<int> & pos()const{return pos_;}

   private:
    const ConstArrayBase * host_;
    const std::vector<int> &dims_;
    std::vector<int> pos_;
  };

}


#endif //  BOOM_ARRAY_ITERATOR_HPP_
