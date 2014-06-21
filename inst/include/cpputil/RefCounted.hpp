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

#ifndef BOOM_REF_COUNTED_HPP
#define BOOM_REF_COUNTED_HPP

#ifndef NO_BOOST_THREADS
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#endif

namespace BOOM{

#ifdef NO_BOOST_THREADS
#define BOOM_LOCK(MUTEX)
#else
#define BOOM_LOCK(MUTEX) boost::lock_guard<boost::mutex> lock(MUTEX)
#endif

  class RefCounted{
    unsigned int cnt_;
#ifndef NO_BOOST_THREADS
    boost::mutex ref_count_mutex_;
#endif
  public:
    RefCounted(): cnt_(0){}
    RefCounted(const RefCounted &)
  : cnt_(0)
#ifndef NO_BOOST_THREADS
  , ref_count_mutex_()
#endif
    {}

    // If this object is assigned a new value, nothing is done to the
    // reference count, so assignment is a no-op.
    RefCounted & operator=(const RefCounted &rhs) { return *this; }

    virtual ~RefCounted(){}
    void up_count(){
      BOOM_LOCK(ref_count_mutex_);
      ++cnt_;
    }
    void down_count(){
      BOOM_LOCK(ref_count_mutex_);
      --cnt_;
    }
    unsigned int ref_count()const{return cnt_;}
  };

#undef BOOM_LOCK
}
#endif // BOOM_REF_COUNTED_HPP
