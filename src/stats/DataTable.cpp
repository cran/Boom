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

#include <cctype>

#include <stats/DataTable.hpp>
#include <cpputil/DefaultVnames.hpp>
#include <cpputil/string_utils.hpp>
#include <cpputil/str2d.hpp>
#include <cpputil/report_error.hpp>
#include <Models/CategoricalData.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <LinAlg/Types.hpp>
#include <cpputil/Ptr.hpp>

namespace BOOM{
  using std::ostringstream;

  typedef std::vector<Ptr<CategoricalData> > CatVec;

  const Vec & get(const std::map<uint, Vec>  &m, uint i){
    return m.find(i)->second;}

  const CatVec & get(const std::map<uint, CatVec>  &m, uint i){
    return m.find(i)->second;}

  inline void field_length_error
  (const string &fname, uint line, uint nfields, uint prev_nfields){
    ostringstream msg;
    msg << "file: " << fname<<endl << " line number "<<line <<" has "
        << nfields <<" fields.  Previous lines had " << prev_nfields
        << "fields." << endl;
    report_error(msg.str());
  }

  //-----------------------------------------------------------------
  inline void wrong_type_error(uint line_num, uint field_num){
    ostringstream msg;
    msg << "line number " << line_num << " field number " << field_num <<endl;
    report_error(msg.str());
  }

  inline void unknown_type(){
    report_error("unknown type");
  }
  //-----------------------------------------------------------------

  typedef std::vector<string> Svec;
  typedef std::vector<bool> BoolVec;

  DataTable::DataTable(const string &fname, bool header, const string &sep){
    ifstream in(fname.c_str());
    if(!in){
      string msg = "bad file name ";
      report_error(msg + fname);
    }

    StringSplitter split(sep);
    string line;
    uint nfields = 0;
    uint line_number=0;

    std::vector<Svec> LabelMap;
    typedef std::vector<double> dVector;
    std::vector<dVector> ContMap;

    if(header){
      ++line_number;
      getline(in, line);
      vnames_ = split(line); }

    while(in){
      ++line_number;
      getline(in, line);
      if(is_all_white(line)) continue;
      std::vector<string> fields = split(line);

      if(nfields== 0){        // getting started
        nfields= fields.size();
        diagnose_types(fields);
        ContMap.resize(nfields);
        LabelMap.resize(nfields);
      }

      if(fields.size() !=  nfields ){  // check number of fields
        field_length_error(fname, line_number, nfields, fields.size());
      }

      for(uint i=0; i<nfields; ++i){
        if(!check_type(vtypes[i], fields[i]))
          wrong_type_error(line_number, i+1);

        if(vtypes[i]==continuous){
          double tmp = str2d(fields[i]);
          ContMap[i].push_back(tmp);
        }else if(vtypes[i]==categorical){
          LabelMap[i].push_back(fields[i]);
        }else{
          unknown_type();}}}


    for(uint i=0; i<nfields; ++i){
      if(vtypes[i] == continuous){
        cont_vars.push_back(Vec(ContMap[i].begin(), ContMap[i].end()));
      }else{
        cont_vars.push_back(Vec(0));
      }
    }

    for(uint i=0; i<nfields; ++i){
      if(vtypes[i] == categorical){
        cat_vars.push_back(make_catdat_ptrs(LabelMap[i]));
      }else{
        std::vector<Ptr<CategoricalData> > tmp;
        cat_vars.push_back(tmp);
      }
    }

    if(vnames_.size()==0) vnames_ =default_vnames(vtypes.size());
  }

  //------------------------------------------------------------

  void DataTable::diagnose_types (const std::vector<string> &vs){
    // determines the type of variable stored in vs

    uint nfields = vs.size();
    vtypes = std::vector<variable_type>(nfields, unknown);
    for(uint i = 0; i<vs.size(); ++i){
      vtypes[i] = is_numeric(vs[i]) ? continuous : categorical;
    }
  }


  bool DataTable::check_type(variable_type t, const string &s)const{
    if(is_numeric(s)){
      if(t==continuous) return true;
    }else{  // s is not numeric
      if(t==categorical) return true;
    }
    return false;
  }

  std::vector<string> & DataTable::vnames(){return vnames_;}
  const std::vector<string> & DataTable::vnames()const{return vnames_;}

  //------------------------------------------------------------
  uint DataTable::nvars()const{ return vtypes.size();}

  LabeledMatrix DataTable::design(bool add_int)const{
    std::vector<bool> include(nvars(),true);
    return design(include, add_int);  }

  //------------------------------------------------------------
  LabeledMatrix DataTable::design
  (const std::vector<bool> &include, bool add_int)const{

    if(include.size()!=nvars())
      report_error("wrong sized include vector in DataTable::design");

    uint n = nobs();         // determine number of rows in design matrix

    //------ determine p: the number of columns in the design matrix
    uint p = add_int ? 1 : 0;   // intercept
    for(uint i = 0; i<nvars(); ++i){
      if(include[i]){
        uint inc = nlevels(i);
        if(vtypes[i]==categorical) --inc; // baseline category
        p+=inc;}}

    Mat X(n,p);
    for(uint i=0; i<n; ++i){   // begin filling matrix
      if(add_int) X(i,0)=1.0;
      uint jj= add_int ? 1 : 0;
      for(uint j=0; j<nvars(); ++j){
        if(include[j]){
          if(vtypes[j]==continuous){
            X(i,jj++) = cont_vars[j][i];
          }else if(vtypes[j]==categorical){
            const Ptr<CategoricalData>  x(cat_vars[j][i]);
            for(uint k =1; k<x->nlevels(); ++k)
              X(i,jj++) = (k==x->value() ? 1:0);
          }else unknown_type(); }}}  //--- done filling matrix

    std::vector<string> dimnames;
    if(add_int) dimnames.push_back("Intercept");
    for(uint j=0; j<nvars(); ++j){
      if(include[j]){
        if(vtypes[j]==continuous)
          dimnames.push_back(vnames_[j]);
        else{
          string stub=vnames_[j];
          const Ptr<CategoricalData> x(cat_vars[j][0]);
          std::vector<string> labs = x->labels();
          for(uint i = 1; i<labs.size(); ++i)
            dimnames.push_back(stub+":"+labs[i]);}}}

    return LabeledMatrix(X, std::vector<std::string>(), dimnames);
  }

  //----------------------------------------------------------------------
  LabeledMatrix DataTable::design
  (std::vector<uint> indx, bool add_int, uint count_from)const{

    uint n=nobs();
    if(count_from>0){
      for(uint i=0; i<indx.size(); ++i){
        indx[i]-= count_from;}}
    uint p = add_int? 1 : 0;
    for(uint i =0; i<indx.size(); ++i){
      uint J = indx[i];
      uint inc =1;
      if(vtypes[J]==categorical) inc = nlevels(J)-1;
      p+=inc;}

    Mat X(n,p);
    for(uint i=0; i<n; ++i){
      if(add_int) X(i,0)=1.0;
      uint jj= add_int ? 1 : 0;
      for(uint j = 0; j<indx.size(); ++j){
        uint J = indx[j];
        if(vtypes[J]==continuous){
          X(i,jj++) = cont_vars[J][i];
        }else if(vtypes[J]==categorical){
          const Ptr<CategoricalData> x(cat_vars[J][i]);
          for(uint k=1; k<x->nlevels();++k)
            X(i,jj++) = (k==x->value() ? 1 : 0);
        }else{
          unknown_type();}}}

    std::vector<string> dimnames;
    if(add_int) dimnames.push_back("Intercept");
    for(uint j=0; j<indx.size(); ++j){
      uint J = indx[j];
      if(vtypes[J]==continuous) dimnames.push_back(vnames_[J]);
      else if(vtypes[J]==categorical){
        const Ptr<CategoricalData> x(cat_vars[J][0]);
        string stub = vnames_[J];
        std::vector<string> labs = x->labels();
        for(uint i=1; i<labs.size(); ++i)
          dimnames.push_back(stub+":"+labs[i]);}}
    return LabeledMatrix(X, std::vector<std::string>(), dimnames);
  }

  //============================================================

  template<class T>
  uint mapsize(const std::map<uint, T> &m){
    if(m.empty()) return 0;
    const T& first_element(m.begin()->second);
    return first_element.size();}


  uint DataTable::nobs()const{
    if(vtypes[0]==continuous) return cont_vars[0].size();
    return cat_vars[0].size();
  }

  uint DataTable::nlevels(uint i)const{
    if(vtypes[i]==continuous) return 1;
    return cat_vars[i][0]->nlevels();
  }

  Vec DataTable::getvar(uint n, uint count_from)const{
    n-= count_from;
    if(vtypes[n]==continuous) return cont_vars[n];
    Vec ans(nobs());
    for(uint i=0; i<nobs(); ++i) ans[i] = cat_vars[n][i]->value();
    return ans; }


  std::vector<Ptr<CategoricalData> >
  DataTable::get_nominal(uint n, uint count_from)const{
    n-= count_from;
    if(vtypes[n]!=categorical) wrong_type_error(1, n);
    return cat_vars[n];}


  std::vector<Ptr<OrdinalData> >
  DataTable::get_ordinal(uint n, uint count_from)const{
    n-= count_from;
    if(vtypes[n]!=categorical) wrong_type_error(1, n);
    std::vector<Ptr<OrdinalData> > ans;
    const std::vector<Ptr<CategoricalData> > &v(cat_vars[n]);

    typedef std::vector<string> Svec;
    typedef boost::shared_ptr<Svec> SVPtr;

    for(uint i=0; i<v.size(); ++i){
      NEW(OrdinalData, dp)(v[i]->value(), v[0]->key());
      ans.push_back(dp);}
    return ans;
  }

  std::vector<Ptr<OrdinalData> >
  DataTable::get_ordinal
  (uint n, const std::vector<string> &ord, uint count_from)const{
    n-= count_from;
    std::vector<Ptr<OrdinalData> > ans(get_ordinal(n));
    set_order(ans, ord);
    return ans;
  }

  //------------------------------------------------------------

  inline ostream &print_cat(ostream &out, const std::vector<Ptr<CategoricalData> > & dv){
    uint n = dv.size();
    for(uint i=0; i<n ; ++i){
      out << dv[i]->value() << " " << dv[i]->lab() << endl;
    }
    return out;
  }

  ostream & DataTable::print(ostream &out, uint from, uint to)const{
    if(to > nobs()) to = nobs();

    uint N = nvars();
    const Svec &vn(vnames());
    std::vector<uint> fw(nvars());
    uint padding = 2;
    for(uint i=0; i<N; ++i) fw[i] = vn[i].size()+padding;

    using std::setw;
    std::vector<Svec> labmat(nvars());
    for(uint j=0; j<nvars(); ++j){
      Svec &v(labmat[j]);
      v.reserve(nobs());
      bool is_cont = vtypes[j]==continuous;
      for(uint i=0; i<nobs(); ++i){
        ostringstream sout;
        if(is_cont) sout << cont_vars[j][i];
        else sout << cat_vars[j][i]->lab();
        string lab = sout.str();
        fw[j] = std::max<uint>(fw[j], lab.size()+padding);
        v.push_back(lab);
      }}

    for(uint j=0; j<nvars(); ++j) out << setw(fw[j]) << vn[j];
    out << endl;

    for(uint i=from; i<to; ++i){
      for(uint j = 0; j<nvars(); ++j){
        out << setw(fw[j]) << labmat[j][i];
      }
      out << endl;
    }
    return out;
  }
  //------------------------------------------------------------
  ostream & operator<<(ostream &out, const DataTable &dt){
    dt.print(out, 0, dt.nobs());
    return out;
  }

}// closes namespace BOOM
