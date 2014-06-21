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

#include <cpputil/RefCounted.hpp>
#include <Models/StateSpace/Filters/ScalarKalmanStorage.hpp>
#include <Models/TimeSeries/TimeSeries.hpp>
#include <Models/DataTypes.hpp>
#include <Models/MvnModel.hpp>

namespace BOOM{
  class ScalarHomogeneousKalmanFilter
    : private RefCounted
  {
   public:
    ScalarHomogeneousKalmanFilter();
    ScalarHomogeneousKalmanFilter(const Vec &Z, double H, const Mat & T,
                                  const Mat &R, const Spd & Q,
                                  Ptr<MvnModel>);

    virtual ScalarHomogeneousKalmanFilter * clone()const;

  // returns log p(ts | theta)
    double fwd(const TimeSeries<DoubleData> &ts);
    double fwd(const Vec & ts);

    Mat  bkwd_sampling();  // returns state
    void bkwd_smoother();

    void set_initial_state_distribution(Ptr<MvnModel> m);
    void set_matrices(const Vec &Z, double H, const Mat & T,
                      const Mat &R, const Spd & Q);
    void impute_state(const TimeSeries<DoubleData> & ts);

    void state_mean_smoother();
    Vec disturbance_smoother(); // returns r0
    Vec disturbance_smoother(std::vector<ScalarKalmanStorage> &g, uint n);
    void state_mean_smoother(std::vector<ScalarKalmanStorage> &g, uint n);
    const Vec & a(uint i)const;

    friend void intrusive_ptr_add_ref(ScalarHomogeneousKalmanFilter *d){
      d->up_count();}
    friend void intrusive_ptr_release(ScalarHomogeneousKalmanFilter *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

   private:
    double update(double y, Vec &a, Spd & P, Vec &K, double &F, double &v,
		  bool missing=false);

    void kalman_smoother_update(Vec &a, Spd &P, const Vec &K, double F);

    // returns p(ts | theta)
    double fwd_vec(const Vec & ts, std::vector<ScalarKalmanStorage> &);

    double initialize(ScalarKalmanStorage &s);

    std::pair<Vec, Mat> simulate_fake_data(uint n);
    Vec simulate_initial_state()const;

    Ptr<MvnModel> initial_state_distribution_;

    Mat T_;
    Vec Z_;
    Mat R_;
    Spd Q_;
    Mat L_;

    Vec r_;
    Spd N_;
    double H_;
    Spd RQR_;
    std::vector<ScalarKalmanStorage> f_;
    uint n_;
  };
}
