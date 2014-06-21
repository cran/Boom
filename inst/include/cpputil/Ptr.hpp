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

#ifndef BOOM_SMART_PTR_H
#define BOOM_SMART_PTR_H

#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#define NEW(T,y) Ptr<T> y = new T
// NEW(very_long_type_name, variable_name)(constructor, arguments)
// is shorthand for
//
// Ptr<very_long_type_name> variable_name = new
// very_long_type_name(constructor, arguments)

namespace BOOM{

  template <class T, bool INTRUSIVE=true> class Ptr;


  //======================================================================
  template <class T>
  class Ptr<T,true>{  // intrusive pointers
    boost::intrusive_ptr<T> pt;
  public:
    typedef T element_type;
    typedef Ptr<T,false> this_type;
    typedef T * this_type::*unspecified_bool_type;

    const boost::intrusive_ptr<T> & get_boost()const{return pt;}
    boost::intrusive_ptr<T> & get_boost(){return pt;}

    Ptr() : pt(){} // never throws
    Ptr(T*p, bool add_ref = true) : pt(p, add_ref){}
    Ptr(const Ptr &rhs) : pt(rhs.pt){}

    template<class Y>
    Ptr(const Ptr<Y,true> &rhs): pt(rhs.get_boost()){}

    ~Ptr(){}  // deletes pt

    Ptr & operator=(const Ptr & rhs){
      if(&rhs !=this) pt = rhs.pt;
      return *this;}
    template<class Y>
    Ptr & operator=(const Ptr<Y,true> &rhs){
      pt = rhs.get_boost();
      return *this;}
    template<class Y>
    Ptr & operator=(T * r){
      pt = r;
      return *this; }

    bool operator<(const Ptr<T,true> & rhs)const{
      return  get() < rhs.get();}

    T & operator*() const{ return *pt;}
    T * operator->() const{ return pt.operator->();}
    T * get() const{return pt.get();}

    template <class U>
    Ptr<U, true> dcast()const{
      return Ptr<U,true>(dynamic_cast<U*>(pt.get()));}

    template <class U>
    Ptr<U, true> scast()const{
      return Ptr<U,true>(static_cast<U*>(pt.get()));}

    template <class U>
    Ptr<U, true> rcast()const{
      return Ptr<U,true>(reinterpret_cast<U*>(pt.get()));}

    template <class U>
    Ptr<U, true> ccast()const{
      return Ptr<U,true>(const_cast<U*>(pt.get()));}


    operator unspecified_bool_type() const{
      return !pt ? 0 : &this_type::get_boost();
    }; // never throws
    bool operator!()const {return !pt;}
    bool operator==(const Ptr &rhs)const{
      return pt==rhs.pt; }
    bool operator!=(const Ptr &rhs)const{
      return pt!=rhs.pt;}

    void swap(Ptr &b){ pt.swap(b.get_boost());}
    void reset(){ pt.reset();}
    void reset(T *new_value){ pt.reset(new_value);}
  };

  //======================================================================
  template <class T>
  class Ptr<T,false>{
    boost::shared_ptr<T> pt;
  public:
    const boost::shared_ptr<T> & get_boost()const {return pt;}
    boost::shared_ptr<T> & get_boost() {return pt;}

    typedef T element_type;
    typedef Ptr<T,false> this_type;

    typedef T * this_type::*unspecified_bool_type;

    Ptr(): pt(){}
    template<class Y> Ptr(Y* p) : pt(p){}
    template<class Y,class D> Ptr(Y* p, D d) : pt(p,d){}
    ~Ptr(){}

    Ptr(const Ptr &rhs): pt(rhs.pt){}
    template <class Y> Ptr(const Ptr<Y> &rhs): pt(rhs.get_boost()){}
    template <class Y> Ptr(std::auto_ptr<Y> &rhs): pt(rhs){}
    explicit Ptr(const boost::shared_ptr<T> &rhs): pt(rhs){}

    Ptr & operator=(const Ptr &rhs){
      if(&rhs==this) return *this;
      pt = rhs.pt;
      return *this; }

    template<class Y>
    Ptr & operator=(const Ptr<Y> &rhs){
      pt = rhs.get_boost();
      return *this; }

    template<class Y>
    Ptr & operator=(const std::auto_ptr<Y> &rhs){
      pt = rhs;
      return *this; }

    template <class Y>
    Ptr & operator=(Y* rhs){   // normal boost pointers prohibit this
      if(pt.get()==rhs) return *this;
      if(!rhs) pt.reset();
      else Ptr(rhs).swap(*this);
      return *this; }

    Ptr & operator=(const boost::shared_ptr<T> & rhs){
      pt = rhs;
      return *this; }

    inline void reset(){ pt.reset();}
    template <class Y> inline void reset(Y *p){ pt.reset(p);}
    template <class Y, class D> inline void reset(Y*p, D d){pt.reset(p,d);}

    bool operator!()const {return !pt;}

    T & operator*()const{return *pt;}
    T * operator->()const{return pt.operator->();}
    T * get() const{return pt.get();}

    bool operator<(const Ptr<T,false> & rhs)const{
      return get() < rhs.get();}

    bool unique()const{return pt.unique();}
    long use_count() const{return pt.use_count();}

    void swap(Ptr &b){ pt.swap(b.get_boost());}

    template <class U>
    Ptr<U,false> dcast()const{
      return Ptr<U,false>(boost::dynamic_pointer_cast<U>(pt));}

    template <class U>
    Ptr<U,false> scast()const{
      return Ptr<U,false>(boost::static_pointer_cast<U>(pt));}

    template <class U>
    Ptr<U,false> ccast()
      const{return Ptr<U,false>(boost::const_pointer_cast<U>(pt));}

    template <class U> friend
    bool operator< (const Ptr &, const Ptr<U,false> &);

    template <class A, class B> friend
    std::basic_ostream<A,B>
    operator<< (std::basic_ostream<A,B> &os, const Ptr&);

    template<class D> friend
    D * get_deleter(const Ptr<T,false> &);

  };

  //-------------------------------------------------------

  template<class T, class U>
  inline
  bool operator==(const Ptr<T,false> &a, const Ptr<U,false> &b)
  {return a.get()==b.get();}

  template<class T, class U>
  inline
  bool operator!=(const Ptr<T,false> &a, const Ptr<U,false> &b){
    return a.get()!=b.get();}

  template <class T, class U>
  inline
  bool operator<(const Ptr<T,false> &a, const Ptr<T,false> &b){
    return a.pt < b.pt;}

  //-------------------------------------------------------
  template<class T>
  inline
  void swap(Ptr<T,false> &a, Ptr<T,false> &b){ a.swap(b);}

  template<class T>
  inline
  T * get_pointer(Ptr<T,false> &a){ return a.get();}

  //-------------------------------------------------------

  template <class A, class B, class T>
  inline
  std::basic_ostream<A,B> &
  operator<< (std::basic_ostream<A,B> &os, const Ptr<T,false> &a){
    os << a.pt;
    return os;}

  template<class T, class D>
  inline
  D * get_deleter(const Ptr<T,false> &a){ return a.pt.get_deleter();}


  template <class To, class From>
  Ptr<To,false> dcast(const Ptr<From,false> &a){
    boost::shared_ptr<To> tmp =
      boost::dynamic_pointer_cast<To>(a.get_boost());
    return  Ptr<To,false>(tmp);
  }

  template <class To, class From>
  Ptr<To,true> dcast(const Ptr<From,true> &a){
    return Ptr<To>( dynamic_cast<To*>(a.get()));}


}
#endif //BOOM_SMART_PTR_H
