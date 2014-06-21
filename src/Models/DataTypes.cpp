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

#include <Models/DataTypes.hpp>
#include <vector>
#include <stdexcept>
#include <sstream>

namespace BOOM{

  ostream & operator<<(ostream &out , const dPtr dp){
    return dp->display(out); }

  ostream & operator<<(ostream & out, const Data &d){
    d.display(out);
    return out;}

  void intrusive_ptr_add_ref(Data *d){
    d->up_count();}
  void intrusive_ptr_release(Data *d){
    d->down_count();
    if(d->ref_count()==0) delete d; }

  Data::missing_status Data::missing()const{
    return missing_flag; }
  void Data::set_missing_status(missing_status m){
    missing_flag=m; }

  //------------------------------------------------------------
  VectorData::VectorData(uint n, double X)
    : x(n,X)
  {}
  VectorData::VectorData(const Vec &y)
    : x(y)
  {}
  VectorData::VectorData(const VectorData &rhs)
    : Data(rhs),
      Traits(rhs),
      x(rhs.x)
  {}
  VectorData * VectorData::clone()const{
    return new VectorData(*this); }

  ostream & VectorData::display(ostream &out)const{
    out << x ; return out;}

  void VectorData::set(const Vec &rhs, bool sig){
    x =rhs;
    if(sig) signal();
  }

  void VectorData::set_element(double value, int position, bool sig){
    x[position] = value;
    if(sig) signal();
  }

  double VectorData::operator[](uint i)const{
    return x[i];}

  double & VectorData::operator[](uint i){
    signal();
    return x[i];
  }

  //------------------------------------------------------------
  MatrixData::MatrixData(int r, int c, double val)
    : x(r, c, val)
  {}

  MatrixData::MatrixData(const Mat &y)
    : Data(),
      x(y)
  {}

  MatrixData::MatrixData(const MatrixData &rhs)
    : Data(rhs),
      Traits(rhs),
      x(rhs.x)
  {}

  MatrixData * MatrixData::clone()const{
    return new MatrixData(*this);}
  ostream & MatrixData::display(ostream &out) const{
    out << x << endl;
    return out;}

  void MatrixData::set(const Mat &rhs, bool sig){
    x = rhs;
    if(sig) signal();
  }

  void MatrixData::set_element(double value, int row, int col, bool sig){
    x(row, col) = value;
    if(sig) signal();
  }

  //------------------------------------------------------------
  CorrData::CorrData(const Spd &y)
    : R(var2cor(y))
  {}

  CorrData::CorrData(const Corr &y)
    : R(y)
  {}

  CorrData::CorrData(const CorrData &rhs)
    : Data(rhs),
      Traits(rhs),
      R(rhs.R)
  {}
  CorrData* CorrData::clone()const{
    return new CorrData(*this);}
  ostream & CorrData::display(ostream &out) const{
    out << R << endl;
    return out;}

  const Corr & CorrData::value()const{return R;}
  void CorrData::set(const Corr &rhs, bool sig){
    R = rhs;
    if(sig) signal();
  }

  //------------------------------------------------------------

  typedef BinomialProcessData BPD;
  typedef std::vector<bool> Vb;

  bool BPD::space_output_(true);
  void BPD::space_output(bool v){ space_output_ = v;}

  BPD::BinomialProcessData(){}
  BPD::BinomialProcessData(const Vb &rhs)
    : dat(rhs)
  {}
  BPD::BinomialProcessData(const BPD &rhs)
    : Data(rhs),
      Traits(rhs),
      dat(rhs.dat)
  {}

  BPD::BinomialProcessData(uint p, bool all)
    : dat(p, all)
  {}
  BPD* BinomialProcessData::clone()const{
    return new BPD(*this);}

  BPD & BPD::operator=(const BPD &rhs){
    if(&rhs!=this){
      dat = rhs.dat;
    }
    return *this;
  }

  bool BPD::operator==(const BPD & rhs)const{
    return dat==rhs.dat; }

  bool BPD::operator==(const Vb & rhs)const{
    return dat==rhs; }

  bool BPD::operator<(const BPD &rhs)const{
    return dat < rhs.dat;  }
  bool BPD::operator>(const BPD &rhs)const{
    return dat > rhs.dat;  }
  bool BPD::operator>=(const BPD &rhs)const{
    return dat >= rhs.dat;  }
  bool BPD::operator<=(const BPD &rhs)const{
    return dat <= rhs.dat;  }

  ostream &BPD::display(ostream &out)const{
    for(uint i = 0; i < dat.size(); ++i){
      out << (dat[i] ? '1' : '0');
      if(space_output_) out << ' ';}
    return out;}

  const Vb & BPD::value()const{ return dat; }
  void BPD::set(const Vb &rhs, bool sig){
    dat = rhs;
    if(sig) signal();
  }

  bool BPD::operator[](uint i)const{return dat[i];}
  void BPD::set_bit(uint i, bool val){ dat[i]=val;}
  void BPD::swap(BPD &d){ std::swap(dat, d.dat); }

  bool BPD::all()const{
    if(std::find(dat.begin(), dat.end(), false)==dat.end())
      return true;
    return false;
  }

  bool BPD::none()const{
    if(std::find(dat.begin(), dat.end(), true)==dat.end())
      return true;
    return false;}

  Vb::const_iterator BPD::begin()const{
    return dat.begin();}
  Vb::const_iterator BPD::end()const{
    return dat.end();}


}// ends namespace BOOM
