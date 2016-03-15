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

#ifndef BOOM_CATEGORICAL_DATA_HPP
#define BOOM_CATEGORICAL_DATA_HPP

#include <Models/DataTypes.hpp>
#include <cpputil/RefCounted.hpp>
#include <stats/FreqDist.hpp>
#include <vector>
#include <set>

namespace BOOM{

  class CategoricalData;
  //------------------------------------------------------------

  class CatKey : public RefCounted{
  // a CatKey should be held by a smart pointer, as it will be shared
  // by many individual CategoricalData.  It contains a set of
  // observers, which must be dumb pointers if CategoricalData are
  // created without using operator new.

  public:
    typedef std::vector<string> StringVector;

    CatKey();
    CatKey(uint Nlev);
    CatKey(const StringVector &Labs);  // sorted list of labels
    CatKey(const CatKey &rhs);        // observers not copied

    void Register(CategoricalData *);
    void Register(CategoricalData *, const string &lab, bool grow=true);
    void Remove(CategoricalData *);
    const string & operator[](uint i)const;
    const StringVector  & labels()const;
    uint findstr(const string &rhs, bool & found)const;
    uint findstr(const string &rhs)const;
    void add_label(const string &lab);  // notifies observers
    uint size()const;
    void reorder(const StringVector &sv);
    void relabel(const StringVector &sv);
    void set_levels(const StringVector &sv);
    ostream & print(ostream &out)const;
  private:
    std::vector<string> labs_;
    std::set<CategoricalData *> observers;
    // Use ordinary pointers... not smart pointers!  Categorical data
    // will register itself with the key using the "this" pointer.  If
    // you use smart pointers in the ObsSet then registration will
    // monkey with the reference count and introduce a memory leak.

    friend void intrusive_ptr_add_ref(CatKey *k){ k->up_count();}
    friend void intrusive_ptr_release(CatKey *k){
      k->down_count(); if(k->ref_count()==0) delete k;}
    std::vector<uint> map_levels(const StringVector &sv)const;
  };

  inline ostream & operator<<(ostream &out, const CatKey &k){
    return k.print(out);}

  Ptr<CatKey> make_catkey(const std::vector<string> &);
  Ptr<CatKey> make_catkey(const std::vector<uint> &, bool full_range=true);

  //------------------------------------------------------------
  class CategoricalData : public DataTraits<uint>{
  public:
    typedef std::vector<string> StringVector;
  private:
    uint val_;
    Ptr<CatKey> labs_;
  protected:
    uint findstr(const string &s);
    uint findstr(const string &s)const;
  public:
    // constructors, assingment, comparison...
    ~CategoricalData() override;
    CategoricalData(uint val, uint Nlevels);
    CategoricalData(uint val, Ptr<CatKey> labs);  // can't grow
    CategoricalData(const string &s, Ptr<CatKey> labs, bool grow=false);
    CategoricalData(uint val, CategoricalData &other);
    CategoricalData(const string &s, CategoricalData &other, bool grow=false);

    CategoricalData(const CategoricalData &rhs);  // share label vector
    CategoricalData * clone()const override;                // share label vector

    void set(const uint & val, bool signal=true) override;
    virtual void set(const string &s, bool signal=true);

    bool operator==(uint rhs)const;
    bool operator==(const string &rhs)const;
    bool operator==(const CategoricalData & rhs)const;

    bool operator!=(uint rhs)const{return ! (*this == rhs);}
    bool operator!=(const string &rhs)const{return ! (*this == rhs);}
    bool operator!=(const CategoricalData & rhs)const{return ! (*this == rhs);}

    //  size querries...........
    uint nlevels()const;  //  'value()' can be 0..nelvels()-1

    // value querries.............
    const uint & value()const override;
    const string &lab()const;
    const std::vector<string> & labels()const;
    void relabel(const StringVector &);
    Ptr<CatKey> key()const{return labs_;}
    bool comparable(const CategoricalData &rhs)const;

    // input-output
    ostream & display(ostream &out)const override;

    void print_key(std::ostream & out)const;
    void print_key(const string & fname)const;

    friend void share_labels(std::vector<Ptr<CategoricalData> > &);
    friend void set_order(std::vector<Ptr<CategoricalData> > &,
                          const StringVector &ord);
    friend class CatKey;
  };
  //------------------------------------------------------------
  class OrdinalData : public CategoricalData{
  public:
    OrdinalData(uint val, uint Nlevels);
    OrdinalData(uint val, Ptr<CatKey> labs);
    OrdinalData(const std::string &s, Ptr<CatKey> labs, bool grow=false);

    OrdinalData(const OrdinalData &rhs);
    OrdinalData * clone()const override;

    bool operator<(const OrdinalData &rhs)const;
    bool operator<=(const OrdinalData &rhs)const;
    bool operator>(const OrdinalData &rhs)const;
    bool operator>=(const OrdinalData &rhs)const;

    bool operator<(uint rhs)const;
    bool operator<=(uint rhs)const;
    bool operator>(uint rhs)const;
    bool operator>=(uint rhs)const;

    bool operator<(const string &rhs)const;
    bool operator<=(const string &rhs)const;
    bool operator>(const string &rhs)const;
    bool operator>=(const string &rhs)const;
  };
  //======================================================================

  class CategoricalFreqDist : public FrequencyDistribution {
   public:
    CategoricalFreqDist(const std::vector<Ptr<CategoricalData> > &data);
  };

  std::vector<Ptr<CategoricalData> >
  make_catdat_ptrs(const std::vector<string> &);
  std::vector<Ptr<CategoricalData> >
  make_catdat_ptrs(const std::vector<uint> &);

  std::vector<Ptr<OrdinalData> > make_ord_ptrs(const std::vector<uint> &);

  void share_labels(std::vector<Ptr<CategoricalData> > &);
  void share_labels(std::vector<Ptr<OrdinalData> > &);

  void set_order(std::vector<Ptr<CategoricalData> > &,
                 const std::vector<string> &);
  void set_order(std::vector<Ptr<OrdinalData> > &,
                 const std::vector<string> &);

  void relabel(std::vector<Ptr<CategoricalData> > &);
  void relabel(std::vector<Ptr<OrdinalData> > &);

}  // namespace BOOM

#endif //BOOM_CATEGORICAL_DATA_HPP
