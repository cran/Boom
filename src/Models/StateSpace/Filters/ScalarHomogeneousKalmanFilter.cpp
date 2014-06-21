/*
  Copyright (C) 2008 Steven L. Scott

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

#include <Models/StateSpace/Filters/ScalarHomogeneousKalmanFilter.hpp>
#include <Models/StateSpace/Filters/KalmanTools.hpp> // scalar_kalman_update
#include <distributions.hpp>
#include <cpputil/nyi.hpp>

namespace BOOM{

  typedef ScalarHomogeneousKalmanFilter SHKF;

  SHKF::ScalarHomogeneousKalmanFilter(){}
  SHKF::ScalarHomogeneousKalmanFilter(const Vec &Z, double H,
                                      const Mat & T, const Mat &R, const Spd & Q,
                                      Ptr<MvnModel> init)
      : initial_state_distribution_(init)
  {
    set_matrices(Z,H,T,R,Q);
  }

  void SHKF::set_initial_state_distribution(Ptr<MvnModel> m){
    initial_state_distribution_=m; }

  void SHKF::set_matrices(const Vec &Z, double H, const Mat & T, const Mat &R, const Spd & Q){
    Z_ = Z;
    H_ = H;
    T_ = T;
    R_ = R;
    Q_ = Q;
    RQR_ = sandwich(R_, Q_);
  }

  SHKF * SHKF::clone()const{return new SHKF(*this);}

  double SHKF::update(double y, Vec &a, Spd &P, Vec & K, double & F, double &v,
		      bool missing){
    // y is y[t]

    // a starts as a[t] and ends as a[t+1]  a[t] = E(alpha[t]| Y^{t-1})
    // P starts as P[t] and ends as P[t+1]  P[t] = V(alpha[t]| Y^{t-1})
    // K is output as K[t]
    // Finv is output as Finv[t]
    // v is output as v[t]

    return scalar_kalman_update(y,a,P,K,F,v,missing, Z_, H_, T_, L_ , RQR_);

//     F = P.Mdist(Z_) + H_;
//     double ans=0;
//     if(!missing){
//       K = T_* (P * Z_);
//       K /= F;
//       double mu = Z_.dot(a);
//       ans =dnorm(y, mu, sqrt(F), true);
//     }else{
//       K = Z_ * 0;
//       v = 0;
//     }

//     a = T_ * a;
//     a += K * v;

//     L_ = T_;
//     L_.add_outer(K, Z_, -1);
//     P = T_ * P.Matrix::multT(L_)  + RQR_;

//     return ans;
  }
  //------------------------------------------------------------
  double SHKF::initialize(ScalarKalmanStorage &f){
    ///////////////////////////////
    double ans=0.0;
    f.a = initial_state_distribution_->mu();
    f.P = initial_state_distribution_->Sigma();
    return ans;
  }

  Vec SHKF::simulate_initial_state()const{
    return initial_state_distribution_->sim(); }


  //------------------------------------------------------------
  double SHKF::fwd(const Vec & ts){ return fwd_vec(ts, f_); }
  //------------------------------------------------------------
  double SHKF::fwd_vec(const Vec &ts, std::vector<ScalarKalmanStorage> &f){
    uint n = ts.size();
    n_ = n;
    if(f.size() < n) f.resize(n);
    double ans = 0;
    Vec a = initial_state_distribution_->mu();
    Spd P = initial_state_distribution_->Sigma();
    for(uint i=0; i<n; ++i){
      ScalarKalmanStorage & s(f[i]);
      s.a = a;
      s.P = P;
      double y = ts[i];
      ans += update(y, a, P, s.K, s.F, s.v);
    }
    return ans;
  }

  //------------------------------------------------------------
  double SHKF::fwd(const TimeSeries<DoubleData> & ts){
    n_ = ts.length();
    if(f_.size() < n_+1) f_.resize(n_+1);

    double ans=0;
    Vec a = initial_state_distribution_->mu();
    Spd P = initial_state_distribution_->Sigma();
    for(uint i=0; i<n_; ++i){
      ScalarKalmanStorage & s(f_[i]);
      s.a = a;
      s.P = P;
      double y = ts[i]->value();
      ans += update(y, a, P, s.K, s.F, s.v);
    }
    return ans;
  }


  //------------------------------------------------------------
  void SHKF::bkwd_smoother(){
    uint p = Z_.size();
    if(r_.size()!=p) r_.resize(p);
    if(N_.nrow()!=p || N_.ncol()!=p) N_ = Spd(p);
    r_ = 0.0;
    N_ = 0.0;
    for(uint i=n_-1; i!=0; --i){
      ScalarKalmanStorage &s(f_[i]);
      scalar_kalman_smoother_update(s.a, s.P,s.K, s.F, s.v,
                                    Z_, T_, r_, N_, L_);
    }
  }

  //------------------------------------------------------------
  Vec SHKF::disturbance_smoother(){ return disturbance_smoother(f_, n_); }

  Vec SHKF::disturbance_smoother(std::vector<ScalarKalmanStorage> &f, uint n){
    // replace K with Durbin and Koopman's r

    Vec r(Z_.size(), 0.0);
    Mat Lstart = T_.t();

    for(int i=n-1; i>=0; --i){
      ScalarKalmanStorage & s(f[i]);
      L_ = Lstart;
      L_.add_outer(Z_, s.K, -1);   // L is the transpose of Durbin and Koopman's L
      s.K = r;
      r = L_ * r  + Z_ * (s.v/s.F);  // now r is r[i-1]
    }
    return r;
  }

  //------------------------------------------------------------
  void SHKF::state_mean_smoother(){ this->state_mean_smoother(f_,n_); }

  void SHKF::state_mean_smoother(std::vector<ScalarKalmanStorage> &f,
                                 uint n){
    Vec r = disturbance_smoother(f,n);
    f[0].a  = initial_state_distribution_->mu() +
        initial_state_distribution_->Sigma() * r;
    for(uint i=1; i<n; ++i){
      const Vec & r(f[i-1].K);
      const Vec & a(f[i-1].a);
      f[i].a = T_*a + RQR_*r;
    }
  }

  //------------------------------------------------------------

  Mat SHKF::bkwd_sampling(){
    state_mean_smoother(f_, n_);  // replaces f_[i].a with E(f_[i].a | ts)

    std::pair<Vec, Mat> ans = simulate_fake_data(n_);
    Vec sim = ans.first;
    Mat alpha = ans.second;
    std::vector<ScalarKalmanStorage> g(f_.begin(), f_.begin() + n_);

    fwd_vec(sim, g);
    // replaces g[i].a with E(g[i].a | sim)
    state_mean_smoother(g, sim.size());

    alpha.row(0) += (f_[0].a - g[0].a);
    for(uint i=1; i<n_; ++i) alpha.row(i) += (f_[i].a - g[i].a);
    return alpha;
  }

  //------------------------------------------------------------

  const Vec & SHKF::a(uint i)const{  return f_[i].a; }

  //------------------------------------------------------------

  std::pair<Vec, Mat> SHKF::simulate_fake_data(uint n){
    uint p = T_.nrow();
    Vec ts(n);
    Mat alpha(n, p);

    Vec a = simulate_initial_state();
    double y = rnorm(Z_.dot(a), sqrt(H_));

    ts[0] = y;
    alpha.row(0) = a;
    Vec zero(Q_.nrow(), 0.0);

    for(uint i=1; i<n; ++i){

      a = T_ * a + R_ * rmvn(zero, Q_);
      y = rnorm(Z_.dot(a), sqrt(H_));
      ts[i] = y;
      alpha.row(i) = a;
    }
    return std::make_pair(ts,alpha);
  }

}
