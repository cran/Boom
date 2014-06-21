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

#include <Models/CategoricalData.hpp>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <set>
#include <cpputil/report_error.hpp>

namespace BOOM{

  using std::find;
  using std::sort;
  using std::copy;
  using std::unique_copy;
  using std::runtime_error;

  typedef CategoricalData CD;
  typedef std::vector<string> StringVec;
  typedef StringVec::iterator SVI;
  typedef StringVec::const_iterator SVcI;
  typedef std::vector<Ptr<CD> > CatVec;
  typedef OrdinalData OD;
  typedef std::vector<Ptr<OD> > OrdVec;
  typedef CD::pKey PK;

  typedef CatKey CK;

  namespace {
    inline string u2str(uint u){
      ostringstream out;
      out << u;
      return out.str();
    }
  }

  CK::CatKey(){}

  CK::CatKey(uint Nlev){
    labs_.reserve(Nlev);
    for(uint s=0; s<Nlev; ++s){
      string lab = u2str(s);
      labs_.push_back(lab);
    }
  }

  CK::CatKey(const StringVec &Labs)
    : labs_(Labs)
  {}

  CK::CatKey(const CatKey &rhs)
    : RefCounted(),
      labs_(rhs.labs_) // observers not copied
  {}


  void CK::Register(CD * dat){
    observers.insert(dat);
    dat->labs_ = this;
    if(dat->value() >= labs_.size()){
      report_error("illegal value passed to CatKey::Register");
    }
  }

  void CK::Register(CD * dat, const string &lab, bool grow){
    observers.insert(dat);
    dat->labs_ = this;
    bool found=true;
    uint i = findstr(lab, found);
    if(found) dat->val_ = i;
    else{
      if(grow){
	add_label(lab);
	dat->val_ = findstr(lab, found);
      }
      else report_error("illegal label passed to CatKey::Register");
    }
  }

  void CK::Remove(CD * dat){ observers.erase(dat); }

  const string & CK::operator[](uint i)const{return labs_[i];}
  const StringVec & CK::labels()const{return labs_;}

  uint CK::findstr(const string & lab, bool & found)const{
    StringVec::const_iterator it = std::find(labs_.begin(), labs_.end(), lab);
    if(it==labs_.end()){
      found=false;
      return labs_.size();
    }
    found=true;
    return distance(labs_.begin(), it);
  }

  uint CK::findstr(const string & lab)const{
    bool found(true);
    uint ans = findstr(lab, found);
    if(!found){
      ostringstream out;
      out << "label " << lab << " not found in CatKey::findstr";
      report_error(out.str());
    }
    return ans;
  }

  void CK::add_label(const string &lab){ labs_.push_back(lab); }

  uint CK::size()const{return labs_.size();}

  void CK::reorder(const StringVec &sv){
    if(labs_==sv) return;
    assert(sv.size()==labs_.size());
    std::vector<uint> new_vals(labs_.size());
    for(uint i=0; i<labs_.size(); ++i){
      string lab = labs_[i];
      for(uint j=0; j<sv.size();++j){
	if(lab==sv[j]){
	  new_vals[i]=j;
	  break;}}}

    for(ObsSet::iterator it = observers.begin();
	it!=observers.end(); ++it){
      (*it)->val_ = new_vals[(*it)->val_];}
    labs_ = sv;
  }

  void CK::relabel(const StringVec &sv){
    if(labs_==sv) return;
    assert(sv.size()==labs_.size());
    std::copy(sv.begin(), sv.end(), labs_.begin());
  }

  ostream & CK::print(ostream & out)const{
    uint nlab = labs_.size();
    for(uint i=0; i<nlab; ++i){
      out << "level " << i << " = " << labs_[i] << std::endl;
    }
    return out;
  }

  void CK::set_levels(const StringVec &sv){
    uint nobs = observers.size();
    if(labs_.size()>0  && nobs >0){
      std::vector<uint> new_pos = map_levels(sv);
      typedef ObsSet::iterator it;
      for(it i= observers.begin();  i!=observers.end();++i){
	CategoricalData * dp = *i;
	uint old = dp->value();
	uint y = new_pos[old];
	dp->set(y);
      }
    }
    labs_ = sv;
  }

  std::vector<uint> CK::map_levels(const StringVec &sv)const{
    std::vector<uint> new_pos(labs_.size());
    for(uint i=0; i<labs_.size(); ++i){
      string s = labs_[i];
      bool found_pos=false;
      for(uint j=0; j<sv.size(); ++j){
	found_pos=false;
	if(sv[j]==s){
	  new_pos[i] = j;
	  found_pos = true;
	  break;
	}
	if(!found_pos){
	  ostringstream err;
	  err << "CatKey::map_levels:  the replacement set of category "
	      << "labels is not a superset of the original labels."
	      << endl
	      << "Could not find level: " << labs_[i]
	      << " in replacement labels." << endl;
	  report_error(err.str());
	}
      }
    }
    return new_pos;
  }

  inline uint findstr2(const string &s, const StringVec &sv){
    SVcI it = find(sv.begin(), sv.end(), s);
    if(it==sv.end()){
      ostringstream out;
      out << "string " << s << " not found in findstr2" << endl;
      throw_exception<runtime_error>(out.str());
    }
    return it-sv.begin();
  }

  uint CD::findstr(const string &s){
    bool found=true;
    return labs_->findstr(s, found); }

  uint CD::findstr(const string &s)const{
    bool found(true);
    return labs_->findstr(s, found);}

  CD::~CategoricalData(){
    labs_->Remove(this);
  }

  CD::CategoricalData(uint val, uint Nlevels)
    : val_(val),
      labs_(new CatKey(Nlevels))
  {
    labs_->Register(this);
  }

  CD::CategoricalData(uint val, PK labs)
    : val_(val),
      labs_(labs)
  {

    if(!!labs_){
      assert( val < labs_->size() &&
	      "too few labels supplied to CategoricalData constructor");
    }
    labs_->Register(this);
  }

  CD::CategoricalData(const string & Lab, PK labs, bool grow)
    : Data(),
      val_(0),
      labs_(labs)
  {
    labs_->Register(this, Lab, grow);
  }

  CD::CategoricalData(uint val, CD &other)
    : Data(),
      val_(val),
      labs_(other.labs_)
  {
    labs_->Register(this);
  }

  CD::CategoricalData(const string &Lab, CD &other, bool grow)
    : Data(),
      val_(),
      labs_(other.labs_)
  {
    labs_->Register(this, Lab, grow);
  }


  CD::CategoricalData(const CD &rhs)
    : Data(rhs),
      Traits(rhs),
      val_(rhs.val_),
      labs_(rhs.labs_)
  {}

  CD * CD::clone()const{return new CD(*this);}
  //  CD * CD::create()const{ return new CD();}

  //------------------------------------------------------------
  void CD::set(const uint & rhs, bool sig){
    if(rhs >= nlevels()){
      ostringstream msg;
      msg << "CategoricalData::operator=... argument " << rhs << " too large ";
      throw_exception<runtime_error>(msg.str());
    }
    val_ = rhs;
    if(sig) signal();
  }

  void CD::set(const string &rhs, bool sig){
    if(!labs_){
      labs_ = new CK();
      labs_->Register(this);
    }
    bool found(true);
    uint i = labs_->findstr(rhs, found);
    if(!found){
      labs_->add_label(rhs);
      i = labs_->findstr(rhs, found);
    }
    val_ = i;
    if(sig) signal();
  }

  //------------------------------------------------------------

  bool CD::operator==(uint rhs)const{ return val_ == rhs;}

  bool CD::operator==(const string &rhs)const{
    return lab()==rhs;}

  bool CD::operator==(const CD & rhs)const{
    return val_==rhs.val_; }


  //------------------------------------------------------------

  uint CD::nlevels()const{ return labs_->size();}
  // note:  labs_ must be set in order for this to work

  const uint & CD::value()const{return val_;}
  const string & CD::lab()const{ return (*labs_)[val_]; }
  const StringVec & CD::labels()const{return labs_->labels();}

  bool CD::comparable(const CD &rhs)const { return labs_ == rhs.labs_;}

  inline void incompat(){
    report_error("comparison between incompatible categorical variables");
  }
  //------------------------------------------------------------
  ostream & CD::display(ostream &out)const{
    out << lab();
    return out;}

  void CD::print_key(ostream &out)const{
    for(uint i=0; i<nlevels(); ++i){
      out << (*labs_)[i] << endl;}}

  void CD::print_key(const string &fname)const{
    ofstream out(fname.c_str());
    print_key(out);}

  //======================================================================

  typedef OrdinalData OD;

  OD::OrdinalData(uint val, uint Nlevels)
    : CD(val, Nlevels)
  {}

  OD::OrdinalData(uint val, PK labs)
    : CD(val, labs)
  {}

  OD::OrdinalData(const string &s, PK labs, bool grow)
    : CD(s, labs, grow)
  {}

  OD::OrdinalData(const OD &rhs)
    : Data(rhs),
      CD(rhs)
  {}

  OD * OD::clone()const{ return new OD(*this);}

  bool OD::operator<(uint rhs)const{ return value() < rhs;}
  bool OD::operator<=(uint rhs)const{ return value() <= rhs;}
  bool OD::operator>(uint rhs)const{ return value() > rhs;}
  bool OD::operator>=(uint rhs)const{ return value() >= rhs;}

  bool OD::operator<(const string &rhs)const{
    uint v = findstr(rhs);
    return value() < v;
  }
  bool OD::operator<=(const string &rhs)const{
    uint v = findstr(rhs);
    return value() <= v;
  }
  bool OD::operator>(const string &rhs)const{
    uint v = findstr(rhs);
    return value() > v;
  }
  bool OD::operator>=(const string &rhs)const{
    uint v = findstr(rhs);
    return value() > v;
  }

  bool OD::operator<(const OD &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() < rhs.value();
  }
  bool OD::operator<=(const OD &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() <= rhs.value();
  }
  bool OD::operator>(const OD &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() > rhs.value();
  }
  bool OD::operator>=(const OD &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() >= rhs.value();
  }

  //======================================================================

  PK make_catkey(const StringVec &sv){
    StringVec tmp(sv);
    sort(tmp.begin(), tmp.end());
    StringVec labs;
    unique_copy(tmp.begin(), tmp.end(), back_inserter(labs));
    return new CK(labs);}


  CatVec make_catdat_ptrs(const StringVec &sv){
    uint n =sv.size();
    PK labs = make_catkey(sv);
    CatVec ans(n);
    for(uint i=0; i<n; ++i) ans[i] = new CD(sv[i], labs);
    return ans;
  }

  PK make_catkey(const std::vector<uint> &iv, bool use_full_range){
    uint n =iv.size();
    if(use_full_range){
      // 0 1 3 => 0 1 2 3, with '2' having a zero count
      uint Max=0;
      for(uint i=0; i<n; ++i) Max = std::max(iv[i], Max);
      PK labs(new CatKey(Max+1));  // 0 to Max
      return labs;
    }else{
      std::map<uint, string> lab_map;
      for(uint i=0; i<n; ++i){
	uint val = iv[i];
	if(lab_map.count(val)==0){
	  lab_map[val] = u2str(val);
	}
      }
      std::vector<string> labs;
      labs.reserve(lab_map.size());
      std::map<uint,string>::iterator it = lab_map.begin();
      while(it!=lab_map.end()){
	labs.push_back(it->second);
	++it;
      }
      return new CatKey(labs);
    }
  }


  CatVec make_catdat_ptrs(const std::vector<uint> &iv){
    uint n = iv.size();
    PK labs = make_catkey(iv);
    CatVec ans(iv.size());
    for(uint i=0; i<n; ++i) ans[i] = new CD(iv[i], labs);
    return ans;
  }

  OrdVec make_ord_ptrs(const std::vector<uint> &iv){
    uint n =iv.size();
    uint Max=0;
    for(uint i=0; i<n; ++i) Max = std::max(iv[i], Max);
    PK labs(new CatKey(Max+1));  // 0 to Max
    OrdVec ans(n);
    for(uint i=0; i<n; ++i) ans[i] = new OD(iv[i], labs);
    return ans;
  }

  //======================================================================

  PK get_labels(const CatVec &cv);

  PK get_labels(const CatVec &dv){
    StringVec labs;
    for(uint i = 0; i<dv.size(); ++i){
      StringVec sv = dv[i]->labels();
      sort(sv.begin(), sv.end());
      StringVec tmp;
      tmp.reserve(labs.size() + sv.size());
      set_union(sv.begin(), sv.end(),
		labs.begin(), labs.end(),
		back_inserter(tmp));
      labs.swap(tmp);
    }
    return new CK(labs);
  }

  void share_labels(CatVec &dv){
    Ptr<CK> key = dv[0]->key();
    for(uint i=0; i<dv.size(); ++i)
      key->Register(dv[i].get());}

  void set_order(CatVec &v, const StringVec &s){
    share_labels(v);
    v[0]->key()->reorder(s);
  }

  void set_order(std::vector<Ptr<OrdinalData> > &v, const StringVec &s){
    std::vector<Ptr<CategoricalData> > tmp(v.begin(), v.end());
    set_order(tmp, s);
  }

  CategoricalFreqDist::CategoricalFreqDist(
      const std::vector<Ptr<CategoricalData> > &data){
    const std::vector<std::string> &labels(data[0]->labels());
    int number_of_categories = labels.size();
    std::vector<int> counts(number_of_categories, 0);
    for (int i = 0; i < data.size(); ++i) {
      ++counts[data[i]->value()];
    }
    reset(counts, labels);
  }

} // namespace BOOM
