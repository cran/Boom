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

#include <LinAlg/Array.hpp>
#include <LinAlg/ArrayIterator.hpp>

namespace BOOM{
    ArrayIterator::ArrayIterator(ArrayBase *host,
                                 const std::vector<int> &starting_position)
        : host_(host),
          dims_(host->dim()),
          pos_(starting_position)
    {}


  ArrayIterator::ArrayIterator(ArrayBase *host)
      : host_(host),
        dims_(host->dim()),
        pos_(std::vector<int>(host->ndim(), 0))
  {}

  double & ArrayIterator::operator*()const{return (*host_)[pos_];}

  ConstArrayIterator::ConstArrayIterator(const ConstArrayBase *host,
                                         const std::vector<int> &starting_position)
        : host_(host),
          dims_(host->dim()),
          pos_(starting_position)
    {}


  ConstArrayIterator::ConstArrayIterator(const ConstArrayBase *host)
      : host_(host),
        dims_(host->dim()),
        pos_(std::vector<int>(host->ndim(), 0))
  {}

  double ConstArrayIterator::operator*()const{return (*host_)[pos_];}

}
