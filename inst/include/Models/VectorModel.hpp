/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#ifndef BOOM_VECTOR_MODEL_HPP
#define BOOM_VECTOR_MODEL_HPP
#include <Models/ModelTypes.hpp>

namespace BOOM{

  // Mix-in model classes that supply logp(Vec);

  class VectorModel
    : virtual public Model{
  public:
    virtual double logp(const Vector &x)const =0;
    VectorModel *clone()const override =0;
    virtual Vector sim()const=0;
  };

  class LocationScaleVectorModel
    : virtual public VectorModel{
  public:
    virtual void set_mu(const Vector &)=0;
    virtual void set_Sigma(const SpdMatrix &)=0;
    virtual void set_siginv(const SpdMatrix &)=0;
    virtual void set_S_Rchol(const Vector &sd, const Matrix &L)=0;

    virtual const Vector & mu() const=0;
    virtual const SpdMatrix & Sigma()const=0;
    virtual const SpdMatrix & siginv() const=0;
    virtual double ldsi()const=0;
  };

  class dVectorModel
    : virtual public VectorModel{
  public:
    virtual double dlogp(const Vector &x, Vector &g)const =0;
    dVectorModel *clone()const override =0;
  };

  class d2VectorModel
    : public dVectorModel{
  public:
    virtual double d2logp(const Vector &x, Vector &g, Matrix &h)const =0;
    d2VectorModel *clone()const override =0;
  };

  class DiffVectorModel
    : public d2VectorModel{
  public:
    double logp(const Vector &x)const override;
    double dlogp(const Vector &x, Vector &g)const override;
    double d2logp(const Vector &x, Vector &g, Matrix &h)const override;
    virtual double Logp(const Vector &x, Vector &g, Matrix &h, uint nd)const=0;
    DiffVectorModel *clone()const override =0;
  };


}
#endif// BOOM_VECTOR_MODEL_HPP
