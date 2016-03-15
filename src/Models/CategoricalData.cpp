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
  typedef std::vector<string> StringVector;

  namespace {
    inline string u2str(uint u){
      ostringstream out;
      out << u;
      return out.str();
    }
  }  // namespace

  CatKey::CatKey(){}

  CatKey::CatKey(uint Nlev){
    labs_.reserve(Nlev);
    for(uint s=0; s<Nlev; ++s){
      string lab = u2str(s);
      labs_.push_back(lab);
    }
  }

  CatKey::CatKey(const StringVector &Labs)
    : labs_(Labs)
  {}

  CatKey::CatKey(const CatKey &rhs)
    : RefCounted(),
      labs_(rhs.labs_) // observers not copied
  {}


  void CatKey::Register(CategoricalData * dat){
    observers.insert(dat);
    dat->labs_ = this;
    if(dat->value() >= labs_.size()){
      report_error("illegal value passed to CatKey::Register");
    }
  }

  void CatKey::Register(CategoricalData * dat, const string &lab, bool grow){
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

  void CatKey::Remove(CategoricalData * dat){ observers.erase(dat); }

  const string & CatKey::operator[](uint i)const{return labs_[i];}
  const StringVector & CatKey::labels()const{return labs_;}

  uint CatKey::findstr(const string & lab, bool & found)const{
    StringVector::const_iterator it = std::find(labs_.begin(), labs_.end(), lab);
    if(it==labs_.end()){
      found=false;
      return labs_.size();
    }
    found=true;
    return distance(labs_.begin(), it);
  }

  uint CatKey::findstr(const string & lab)const{
    bool found(true);
    uint ans = findstr(lab, found);
    if(!found){
      ostringstream out;
      out << "label " << lab << " not found in CatKey::findstr";
      report_error(out.str());
    }
    return ans;
  }

  void CatKey::add_label(const string &lab){ labs_.push_back(lab); }

  uint CatKey::size()const{return labs_.size();}

  void CatKey::reorder(const StringVector &sv){
    if(labs_==sv) return;
    assert(sv.size()==labs_.size());
    std::vector<uint> new_vals(labs_.size());
    for(uint i=0; i<labs_.size(); ++i){
      string lab = labs_[i];
      for(uint j=0; j<sv.size();++j){
        if(lab==sv[j]){
          new_vals[i]=j;
          break;}}}

    for(std::set<CategoricalData *>::iterator it = observers.begin();
        it!=observers.end(); ++it){
      (*it)->val_ = new_vals[(*it)->val_];}
    labs_ = sv;
  }

  void CatKey::relabel(const StringVector &sv){
    if(labs_==sv) return;
    assert(sv.size()==labs_.size());
    std::copy(sv.begin(), sv.end(), labs_.begin());
  }

  ostream & CatKey::print(ostream & out)const{
    uint nlab = labs_.size();
    for(uint i=0; i<nlab; ++i){
      out << "level " << i << " = " << labs_[i] << std::endl;
    }
    return out;
  }

  void CatKey::set_levels(const StringVector &sv){
    uint nobs = observers.size();
    if(labs_.size()>0  && nobs >0){
      std::vector<uint> new_pos = map_levels(sv);
      typedef std::set<CategoricalData *>::iterator it;
      for(it i= observers.begin();  i!=observers.end();++i){
        CategoricalData * dp = *i;
        uint old = dp->value();
        uint y = new_pos[old];
        dp->set(y);
      }
    }
    labs_ = sv;
  }

  std::vector<uint> CatKey::map_levels(const StringVector &sv)const{
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

  inline uint findstr2(const string &s, const StringVector &sv){
    std::vector<string>::const_iterator it = find(sv.begin(), sv.end(), s);
    if(it==sv.end()){
      ostringstream out;
      out << "string " << s << " not found in findstr2" << endl;
      report_error(out.str());
    }
    return it-sv.begin();
  }

  uint CategoricalData::findstr(const string &s){
    bool found=true;
    return labs_->findstr(s, found); }

  uint CategoricalData::findstr(const string &s)const{
    bool found(true);
    return labs_->findstr(s, found);}

  CategoricalData::~CategoricalData(){
    labs_->Remove(this);
  }

  CategoricalData::CategoricalData(uint val, uint Nlevels)
    : val_(val),
      labs_(new CatKey(Nlevels))
  {
    labs_->Register(this);
  }

  CategoricalData::CategoricalData(uint val, Ptr<CatKey> labs)
    : val_(val),
      labs_(labs)
  {

    if(!!labs_){
      assert( val < labs_->size() &&
              "too few labels supplied to CategoricalData constructor");
    }
    labs_->Register(this);
  }

  CategoricalData::CategoricalData(const string & Lab,
                                   Ptr<CatKey> labs,
                                   bool grow)
    : Data(),
      val_(0),
      labs_(labs)
  {
    labs_->Register(this, Lab, grow);
  }

  CategoricalData::CategoricalData(uint val, CategoricalData &other)
    : Data(),
      val_(val),
      labs_(other.labs_)
  {
    labs_->Register(this);
  }

  CategoricalData::CategoricalData(const string &Lab,
                                   CategoricalData &other,
                                   bool grow)
    : Data(),
      val_(),
      labs_(other.labs_)
  {
    labs_->Register(this, Lab, grow);
  }


  CategoricalData::CategoricalData(const CategoricalData &rhs)
    : Data(rhs),
      Traits(rhs),
      val_(rhs.val_),
      labs_(rhs.labs_)
  {}

  CategoricalData * CategoricalData::clone()const{
    return new CategoricalData(*this);
  }

  //------------------------------------------------------------
  void CategoricalData::set(const uint & rhs, bool sig){
    if(rhs >= nlevels()){
      ostringstream msg;
      msg << "CategoricalData::operator=... argument " << rhs << " too large ";
      report_error(msg.str());
    }
    val_ = rhs;
    if(sig) signal();
  }

  void CategoricalData::set(const string &rhs, bool sig){
    if(!labs_){
      labs_ = new CatKey();
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

  bool CategoricalData::operator==(uint rhs)const{ return val_ == rhs;}

  bool CategoricalData::operator==(const string &rhs)const{
    return lab()==rhs;}

  bool CategoricalData::operator==(const CategoricalData & rhs)const{
    return val_==rhs.val_; }


  //------------------------------------------------------------

  uint CategoricalData::nlevels()const{ return labs_->size();}
  // note:  labs_ must be set in order for this to work

  const uint & CategoricalData::value()const{return val_;}
  const string & CategoricalData::lab()const{ return (*labs_)[val_]; }
  const StringVector & CategoricalData::labels()const{return labs_->labels();}

  bool CategoricalData::comparable(const CategoricalData &rhs) const {
    return labs_ == rhs.labs_;
  }

  inline void incompat(){
    report_error("comparison between incompatible categorical variables");
  }
  //------------------------------------------------------------
  ostream & CategoricalData::display(ostream &out)const{
    out << lab();
    return out;}

  void CategoricalData::print_key(ostream &out)const{
    for(uint i=0; i<nlevels(); ++i){
      out << (*labs_)[i] << endl;}}

  void CategoricalData::print_key(const string &fname)const{
    ofstream out(fname.c_str());
    print_key(out);}

  //======================================================================


  OrdinalData::OrdinalData(uint val, uint Nlevels)
    : CategoricalData(val, Nlevels)
  {}

  OrdinalData::OrdinalData(uint val, Ptr<CatKey> labs)
    : CategoricalData(val, labs)
  {}

  OrdinalData::OrdinalData(const string &s, Ptr<CatKey> labs, bool grow)
    : CategoricalData(s, labs, grow)
  {}

  OrdinalData::OrdinalData(const OrdinalData &rhs)
    : Data(rhs),
      CategoricalData(rhs)
  {}

  OrdinalData * OrdinalData::clone()const{ return new OrdinalData(*this);}

  bool OrdinalData::operator<(uint rhs)const{ return value() < rhs;}
  bool OrdinalData::operator<=(uint rhs)const{ return value() <= rhs;}
  bool OrdinalData::operator>(uint rhs)const{ return value() > rhs;}
  bool OrdinalData::operator>=(uint rhs)const{ return value() >= rhs;}

  bool OrdinalData::operator<(const string &rhs)const{
    uint v = findstr(rhs);
    return value() < v;
  }
  bool OrdinalData::operator<=(const string &rhs)const{
    uint v = findstr(rhs);
    return value() <= v;
  }
  bool OrdinalData::operator>(const string &rhs)const{
    uint v = findstr(rhs);
    return value() > v;
  }
  bool OrdinalData::operator>=(const string &rhs)const{
    uint v = findstr(rhs);
    return value() > v;
  }

  bool OrdinalData::operator<(const OrdinalData &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() < rhs.value();
  }
  bool OrdinalData::operator<=(const OrdinalData &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() <= rhs.value();
  }
  bool OrdinalData::operator>(const OrdinalData &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() > rhs.value();
  }
  bool OrdinalData::operator>=(const OrdinalData &rhs)const{
    if(!comparable(rhs)) incompat();
    return value() >= rhs.value();
  }

  //======================================================================

  Ptr<CatKey> make_catkey(const StringVector &sv){
    StringVector tmp(sv);
    sort(tmp.begin(), tmp.end());
    StringVector labs;
    unique_copy(tmp.begin(), tmp.end(), back_inserter(labs));
    return new CatKey(labs);}


  std::vector<Ptr<CategoricalData> > make_catdat_ptrs(const StringVector &sv){
    uint n =sv.size();
    Ptr<CatKey> labs = make_catkey(sv);
    std::vector<Ptr<CategoricalData> > ans(n);
    for(uint i=0; i<n; ++i) ans[i] = new CategoricalData(sv[i], labs);
    return ans;
  }

  Ptr<CatKey> make_catkey(const std::vector<uint> &iv, bool use_full_range){
    uint n =iv.size();
    if(use_full_range){
      // 0 1 3 => 0 1 2 3, with '2' having a zero count
      uint Max=0;
      for(uint i=0; i<n; ++i) Max = std::max(iv[i], Max);
      Ptr<CatKey> labs(new CatKey(Max+1));  // 0 to Max
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


  std::vector<Ptr<CategoricalData> > make_catdat_ptrs(const std::vector<uint> &iv){
    uint n = iv.size();
    Ptr<CatKey> labs = make_catkey(iv);
    std::vector<Ptr<CategoricalData> > ans(iv.size());
    for(uint i=0; i<n; ++i) ans[i] = new CategoricalData(iv[i], labs);
    return ans;
  }

  std::vector<Ptr<OrdinalData> > make_ord_ptrs(const std::vector<uint> &iv){
    uint n =iv.size();
    uint Max=0;
    for(uint i=0; i<n; ++i) Max = std::max(iv[i], Max);
    Ptr<CatKey> labs(new CatKey(Max+1));  // 0 to Max
    std::vector<Ptr<OrdinalData> > ans;
    ans.reserve(n);
    for(uint i=0; i<n; ++i) {
      ans.push_back(new OrdinalData(iv[i], labs));
    }
    return ans;
  }

  //======================================================================

  Ptr<CatKey> get_labels(const std::vector<Ptr<CategoricalData> > &cv);

  Ptr<CatKey> get_labels(const std::vector<Ptr<CategoricalData> > &dv){
    StringVector labs;
    for(uint i = 0; i<dv.size(); ++i){
      StringVector sv = dv[i]->labels();
      sort(sv.begin(), sv.end());
      StringVector tmp;
      tmp.reserve(labs.size() + sv.size());
      set_union(sv.begin(), sv.end(),
                labs.begin(), labs.end(),
                back_inserter(tmp));
      labs.swap(tmp);
    }
    return new CatKey(labs);
  }

  void share_labels(std::vector<Ptr<CategoricalData> > &dv){
    Ptr<CatKey> key = dv[0]->key();
    for(uint i=0; i<dv.size(); ++i)
      key->Register(dv[i].get());}

  void set_order(std::vector<Ptr<CategoricalData> > &v, const StringVector &s){
    share_labels(v);
    v[0]->key()->reorder(s);
  }

  void set_order(std::vector<Ptr<OrdinalData> > &v, const StringVector &s){
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
