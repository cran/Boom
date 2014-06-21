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

#ifndef BOOM_TIME_SERIES_BASE_CLASS_HPP
#define BOOM_TIME_SERIES_BASE_CLASS_HPP

#include <sstream>
#include <vector>
#include <Models/DataTypes.hpp>
#include <Models/TimeSeries/MarkovLink.hpp>
#include  <boost/type_traits/is_base_of.hpp>

namespace BOOM{
  //======================================================================
  template<class D>
  bool linked(Ptr<D> d){
    // true if either prev or next is set
    bool linkable = boost::is_base_of<MarkovLink<D>, D>::value;
    if(!linkable) return false;
    return (!!d->next() || !!d->prev());
  }

  //======================================================================

  template <class D>
  class TimeSeries : virtual public Data,
                     public std::vector<Ptr<D> >
  {
    // A TimeSeries is a vector of pointers to data
  public:
    typedef D data_point_type;
    typedef TimeSeries<D> ts_type;

    TimeSeries(const string & ID="");
    TimeSeries(const D&, const string & ID="");
    TimeSeries(const Ptr<D> &, const string & ID="");
    TimeSeries(const std::vector<Ptr<D> > &v, bool reset_links=true,
                const string & ID="");
    template<class FwdIt>
    TimeSeries(FwdIt Beg, FwdIt End, bool reset_links = true,
                bool copy_data=false, const string & ID="");

    TimeSeries(const TimeSeries &);   // value semantics
    TimeSeries<D> * clone()const;      // value semantics
    TimeSeries<D> & operator=(const TimeSeries &rhs);    // copies pointers

    TimeSeries<D> & unique_copy(const TimeSeries & rhs); // clones pointers

    template<class FwdIt>
    TimeSeries<D> & ref(FwdIt Beg, FwdIt End);
    // ref simply copies the pointers without setting links

    void set_links();
    ostream & display(ostream &)const;
    //    istream & read(istream &in);
    uint element_size(bool minimal=true)const;   // size of one data point
    uint length()const;                          // length of the series
    virtual uint size(bool minimal=true)const;

    const string & id()const;
    void set_id(const string &newid);

    // adding data to the time series... add_1 and add_series
    // assimilate their entries so that the time series is remains
    // contiguous, with each element pointing to the next and previous
    // elements.

    // just_add simply adds a pointer to the end of the time series,
    // which may or may not refer to other elements already in the
    // series.
    virtual void add_1(const Ptr<D> &);
    virtual void add_series(const Ptr<TimeSeries<D> > &);
    void just_add(const Ptr<D> &);    // dont bother with links

    void one_line_display(bool d=true);
    void one_line_input(bool i=true);
    void clear();

    private:
    typedef std::vector<Ptr<D> > vec_t;

    Ptr<D> prototype_data_point;
    bool one_line_display_;
    bool one_line_input_;
    string series_id;

    void clone_series(const TimeSeries &);
    // makes *this a copy of rhs.  Copies underlying data and sets links
  };


  typedef TimeSeries<DoubleData> ScalarTimeSeries;
  inline Ptr<ScalarTimeSeries> make_ts(const Vec &y){
    uint n = y.size();
    std::vector<Ptr<DoubleData> > ts;
    ts.reserve(n);
    for(uint i=0; i<n; ++i){
      NEW(DoubleData, yi)(y[i]);
      ts.push_back(yi);
    }
    NEW(ScalarTimeSeries, ans)(ts);
    return ans;
  }

  //======================================================================
  // The structs defined below are used to implment the TimeSeries
  // templates using template meta-programming.  The main problem
  // being solved is whether or not the data elements in the time
  // series have links back and forth.  If they do then the TimeSeries
  // will set those links as data elements are added.

  template <class D, class T>
  struct time_series_data_adder{
    void operator()(Ptr<D>, Ptr<D>){}
  };

  template <class D>
  struct time_series_data_adder<D, boost::true_type>{
    void operator()(Ptr<D> last, Ptr<D> d){
      if(linked(d)) return;  // ??????????????????????????
      if(!last->next()) last->set_next(d);
      if(!d->prev()) d->set_prev(last);
    }
  };

  template <class D>
  struct is_linkable : public boost::is_base_of<MarkovLink<D>, D>{};


  template <class D, class T = is_linkable<D> >
  struct time_series_link_clearer{ void operator()(Ptr<D>){} };

  template <class D>
  struct time_series_link_clearer<D, boost::true_type>{
    void operator()(Ptr<D> d){ d->clear_links(); }  };

  template <class D, class T>
  struct set_links_impl{
    void operator()(std::vector<Ptr<D> > &){}
  };

  template<class D>
  struct set_links_impl<D, boost::true_type>{
    void operator()(std::vector<Ptr<D> > & v){
      uint n = v.size();
      if(n==0) return;
      for(uint i=0; i<n; ++i){
        if(i>0) v[i]->set_prev(v[i-1]);
        if(i<n-1) v[i]->set_next(v[i+1]);
      }
      v.front()->unset_prev();
      v.back()->unset_next();
    }
  };

  //======================================================================

  template<class D>
  TimeSeries<D>::TimeSeries(const string &ID)
    : Data(),
      vec_t(),
      one_line_display_(false),
      one_line_input_(false),
      series_id(ID)
  {

  }

  template<class D>
  TimeSeries<D>::TimeSeries(const D &d, const string &ID)
    : prototype_data_point(d.clone()),
      one_line_display_(false),
      one_line_input_(false),
      series_id(ID)
  {}

  template<class D>
  TimeSeries<D>::TimeSeries(const Ptr<D> &d, const string &ID)
    : prototype_data_point(d->clone()),
      one_line_display_(false),
      one_line_input_(false),
      series_id(ID)
  {}


  template <class D>
  template <class FwdIt>
  TimeSeries<D>::TimeSeries(FwdIt Beg, FwdIt End, bool reset_links,
                              bool , const string &ID)
    : vec_t(Beg,End),
      prototype_data_point(*Beg),
      one_line_display_(false),
      one_line_input_(false),
      series_id(ID)
  {
    if(reset_links) set_links();
  }

  template<class D>
  void TimeSeries<D>::clone_series(const TimeSeries<D> &rhs)
  {
    uint n = rhs.length();
    vec_t::resize(n);
    for(uint i=0; i<n; ++i) (*this)[i] = rhs[i]->clone();
    set_links();
  }



  template <class D>
  void TimeSeries<D>::set_links(){
    typedef typename is_linkable<D>::type isLinked;
    set_links_impl<D, isLinked> impl;
    impl(*this);
  }

  template<class D>
  TimeSeries<D>::TimeSeries(const vec_t &v, bool reset_links, const string &ID)
    : vec_t(v),                       // copies pointers
      prototype_data_point(v.back()),
      one_line_display_(false),
      one_line_input_(false),
      series_id(ID)
  {
    if(reset_links) set_links();
  }

  template<class D>
  TimeSeries<D>::TimeSeries(const TimeSeries &rhs)
    : Data(),
      vec_t(),
      prototype_data_point(rhs.prototype_data_point),
      one_line_display_(rhs.one_line_display_),
      one_line_input_(rhs.one_line_input_),
      series_id(rhs.series_id)
  {
    if(!!prototype_data_point){
      prototype_data_point = rhs.prototype_data_point->clone();
      time_series_link_clearer<D> clear_links;
      clear_links(prototype_data_point);
    }
    clone_series(rhs);
  }


  template<class D>
  TimeSeries<D> & TimeSeries<D>::operator=(const TimeSeries<D> &rhs){
    if(&rhs==this) return *this;
    //changed 10/21/2005.  No longer clones underlying data
    //    clone_series(rhs);
    vec_t::operator=(rhs);     // now just the pointers are copied

    if(!!rhs.prototype_data_point)
      prototype_data_point = rhs.prototype_data_point->clone();
    one_line_display_ = rhs.one_line_display_;
    one_line_input_ = rhs.one_line_input_;
    series_id = rhs.series_id;
    return *this;
  }

  template<class D>
  TimeSeries<D> & TimeSeries<D>::unique_copy(const TimeSeries<D> &rhs){
    if(&rhs==this) return *this;
    clone_series(rhs);

    if(!!rhs.prototype_data_point)
      prototype_data_point = rhs.prototype_data_point->clone();
    one_line_display_ = rhs.one_line_display_;
    one_line_input_ = rhs.one_line_input_;
    series_id = rhs.series_id;
    return *this;
  }

  template<class D>
  template<class FwdIt>
  TimeSeries<D> & TimeSeries<D>::ref(FwdIt Beg, FwdIt End){
    vec_t::assign(Beg,End);
    prototype_data_point = *Beg;
    one_line_display_ = false;
    one_line_input_ = false;
    return *this;
  }

  template<class D>
  TimeSeries<D> * TimeSeries<D>::clone()const{
    return new TimeSeries<D>(*this);}

  template<class D>
  ostream & TimeSeries<D>::display(ostream &out)const{
    for(uint i = 0; i<length(); ++i){
      (*this)[i]->display(out);
      if(one_line_display_) out << " ";
      else out << endl;
    }
    return out;}

  std::vector<string> split_string(const string &s);

//   template<class D>
//   istream & TimeSeries<D>::read(istream &in){
//     if(one_line_input_){
//       string line;
//       getline(in, line);
//       std::vector<string> fields = split_string(line);
//       for(uint i = 0; i<fields.size(); ++i){
//      istringstream tmp_in(fields[i]);
//      Ptr<D> tmp = (length()>0)
//        ? vec_t::back()->clone()
//        : prototype_data_point->clone();
//      tmp->read(tmp_in);
//      add_1(tmp);
//      prototype_data_point = tmp;
//       }
//     }else{
//       while(in){
//      Ptr<D> tmp = (length()==0)
//        ? vec_t::back()->clone()
//        : prototype_data_point->clone();
//      tmp->read(in);
//      if(!in) break;
//      add_1(tmp); }}
//     return in;
//   }

  template<class D>
  uint TimeSeries<D>::element_size(bool minimal)const{
    if(this->empty()) return 0;
    return vec_t::back()->size(minimal); }

  template<class D>
  uint TimeSeries<D>::size(bool)const{ return vec_t::size();}

  template<class D>
  uint TimeSeries<D>::length()const{ return vec_t::size(); }

  template<class D>
  void TimeSeries<D>::set_id(const string &ID){ series_id = ID; }

  template<class D>
  const string & TimeSeries<D>::id()const{ return series_id;}



  template<class D>
  void TimeSeries<D>::add_1(const Ptr<D> &d){
    if(!prototype_data_point){
      prototype_data_point = d->clone();
      time_series_link_clearer<D> clear_links;
      clear_links(prototype_data_point);
    }
    if(length()>0){
      Ptr<D> last = vec_t::back();
      time_series_data_adder<D, is_linkable<D> > adder;
      adder(last, d);
    }
    just_add(d);
  }

  template<class D>
  void TimeSeries<D>::add_series(const Ptr<TimeSeries<D> > &d){
    for(uint i = 0; i<d->length(); ++i) add_1((*d)[i]);  }

  template<class D>
  void TimeSeries<D>::just_add(const Ptr<D> &d){ vec_t::push_back(d); }

  template<class D>
  void TimeSeries<D>::one_line_display(bool d){
    one_line_display_ = d; }

  template<class D>
  void TimeSeries<D>::one_line_input(bool i){
    one_line_input_ = i; }

  template<class D>
  void TimeSeries<D>::clear(){
    if(this->length()>0) prototype_data_point = vec_t::back()->clone();
    bool linkable = is_linkable<D>::value;
    if(linkable){
      if(!!prototype_data_point){
        time_series_link_clearer<D> clear_links;
        clear_links(prototype_data_point); }}
    vec_t::clear();  }
}



#endif // BOOM_TIME_SERIES_BASE_CLASS_HPP
