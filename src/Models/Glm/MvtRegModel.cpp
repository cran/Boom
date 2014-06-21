/*
  Copyright (C) 2007 Steven L. Scott

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
#include <Models/Glm/MvtRegModel.hpp>
#include <LinAlg/QR.hpp>
#include <distributions.hpp>
#include <cpputil/nyi.hpp>

namespace BOOM{
  typedef MvtRegModel MVTR;

  namespace{
    double default_df(30.0);
  }

  MVTR::MvtRegModel(uint xdim, uint ydim)
    : ParamPolicy(new MatrixParams(xdim,ydim),
		  new SpdParams(ydim),
		  new UnivParams(default_df))
  {}

  MVTR::MvtRegModel(const Mat &X,const Mat &Y, bool add_intercept)
    : ParamPolicy(new MatrixParams(X.ncol() + add_intercept,Y.ncol()),
		  new SpdParams(Y.ncol()),
		  new UnivParams(default_df))
  {
    Mat XX(add_intercept? cbind(1.0,X) : X);
    QR qr(XX);
    Mat Beta(qr.solve(qr.QtY(Y)));
    Mat resid = Y - XX* Beta;
    uint n = XX.nrow();
    Spd Sig = resid.t() * resid/n;

    set_Beta(Beta);
    set_Sigma(Sig);

    for(uint i=0; i<n; ++i){
      Vec y = Y.row(i);
      Vec x = XX.row(i);
      NEW(MvRegData, dp)(y,x);
      DataPolicy::add_data(dp);
    }
  }

  MVTR::MvtRegModel(const Mat &B, const Spd &Sigma, double nu)
    : ParamPolicy(new MatrixParams(B),
		  new SpdParams(Sigma),
		  new UnivParams(nu))
  {}

  MVTR::MvtRegModel(const MvtRegModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      LoglikeModel(rhs)
  {}

  MVTR * MVTR::clone()const{ return new MVTR(*this);}

  uint MVTR::xdim()const{ return Beta().nrow();}
  uint MVTR::ydim()const{ return Beta().ncol();}

  const Mat & MVTR::Beta()const{ return Beta_prm()->value();}
  const Spd & MVTR::Sigma()const{return Sigma_prm()->var();}
  const Spd & MVTR::Siginv()const{return Sigma_prm()->ivar();}
  double MVTR::ldsi()const{return Sigma_prm()->ldsi();}
  double MVTR::nu()const{return Nu_prm()->value();}

  Ptr<MatrixParams> MVTR::Beta_prm(){ return ParamPolicy::prm1();}
  Ptr<SpdParams> MVTR::Sigma_prm(){ return ParamPolicy::prm2();}
  Ptr<UnivParams> MVTR::Nu_prm(){ return ParamPolicy::prm3();}
  const Ptr<MatrixParams> MVTR::Beta_prm()const{ return ParamPolicy::prm1();}
  const Ptr<SpdParams> MVTR::Sigma_prm()const{ return ParamPolicy::prm2();}
  const Ptr<UnivParams> MVTR::Nu_prm()const{ return ParamPolicy::prm3();}

  void MVTR::set_Beta(const Mat & B){ Beta_prm()->set(B); }
  void MVTR::set_Sigma(const Spd &V){Sigma_prm()->set_var(V);}
  void MVTR::set_Siginv(const Spd &iV){Sigma_prm()->set_ivar(iV);}
  void MVTR::set_nu(double new_nu){Nu_prm()->set(new_nu);}

  void MVTR::mle(){  // ECME
    nyi("MvtRegModel::mle");
  }

  double MVTR::loglike()const{
    const DatasetType & d(dat());
    uint n = d.size();
    double ans =0;
    for(uint i=0; i<n; ++i) ans += pdf(d[i], true);
    return ans;
  }

  double MVTR::pdf(dPtr dp, bool logscale)const{
    Ptr<DataType> d = DAT(dp);
    const Vec &y(d->y());
    const Vec &X(d->x());
    double ans = dmvt(y, X*Beta(), Siginv(), nu(), ldsi(), true);
    return logscale ? ans : exp(ans);
  }

  Vec MVTR::predict(const Vec &x)const{ return x*Beta(); }

  MvRegData * MVTR::simdat()const{
    Vec x = simulate_fake_x();
    return this->simdat(x);
  }

  MvRegData * MVTR::simdat(const Vec &x)const{
    Vec Y = rmvt(predict(x), Sigma(), nu());
    return new MvRegData(Y,x);
  }

  Vec MVTR::simulate_fake_x()const{
    uint p = xdim();
    Vec x(p);
    x[0] = 1.0;
    for(uint i=0; i<p; ++i) x[i] = rnorm();
    return x;
  }


}
