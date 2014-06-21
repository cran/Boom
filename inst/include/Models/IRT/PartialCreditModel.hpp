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
#ifndef BOOM_PARTIAL_CREDIT_MODEL_HPP
#define BOOM_PARTIAL_CREDIT_MODEL_HPP

#include <Models/IRT/Item.hpp>
#include <Models/IRT/Subject.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Policies/ParamPolicy_3.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/ConstrainedVectorParams.hpp>
namespace BOOM{
  namespace IRT{

    class PcrBetaConstraint : public VectorConstraint{
    public:
      // constrains the first and next_to_last elements of b so that (M+1)*b[0] = b[M]
      // This class is used when PartialCreditModel is kept identified.
      virtual bool check(const Vec &b)const;
      virtual void impose(Vec &b)const;
      virtual Vec expand(const Vec &b_min)const;   // adds in b0
      virtual Vec reduce(const Vec &b_full)const;  // omits b0
    };

    class PcrDConstraint : public VectorConstraint{
    public:
      virtual bool check(const Vec &d)const;
      virtual void impose(Vec &d)const;
      virtual Vec expand(const Vec &d_min)const;
      virtual Vec reduce(const Vec &d_full)const;
    };

    class PartialCreditModel
      : public Item,  // knows all subjects assigned to this item
	public ParamPolicy_3<UnivParams, UnivParams, ConstrainedVectorParams>, // a,b,d
	public PriorPolicy
    {

      /*------------------------------------------------------------
	An item with maxscore()==M yields log score probabilities = C
	+ X*beta where C is a normalizing constant X[0..M, 0..M] is an
	(M+1)x(M+2) matrix and beta[0..M+1] is an M+2 vector as follows
	(for M==4)

	X:                             beta:
	1  0  0  0  0  theta           a*(d0-b)
	0  1  0  0  0  2*theta         a*(d0+d1-2b)
	0  0  1  0  0  3*theta         a*(d0+d1+d2-3b)
	0  0  0  1  0  4*theta         a*(d0+d1+d2+d3-4b)
	0  0  0  0  1  5*theta         a*(-5b)  // sum of d's is 0
				       a

        The redundant information is stored in d, so d[0] = 0 and
	d.sum()=0.  Among other things this makes parameter expansion
	easy.

	------------------------------------------------------------*/

    public:
      PartialCreditModel(const string & Id, uint Mscore, uint which_sub,
			 uint Nscales, const string &Name="", bool id_d0=true);
      PartialCreditModel(const string & Id, uint Mscore, uint which_sub,
			 uint Nscales, double a, double b, const Vec &d,
			 const string &Name="", bool id_d0=true);
      PartialCreditModel(const PartialCreditModel &rhs);
      PartialCreditModel * clone()const;

      uint which_subscale()const;

      Ptr<UnivParams> A_prm(bool check=true);
      Ptr<UnivParams> B_prm(bool check=true);
      Ptr<ConstrainedVectorParams> D_prm(bool check=true);
      Ptr<ConstrainedVectorParams> Beta_prm(bool check=true);
      const Ptr<UnivParams> A_prm(bool check=true)const;
      const Ptr<UnivParams> B_prm(bool check=true)const;
      const Ptr<ConstrainedVectorParams> D_prm(bool check=true)const;
      const Ptr<ConstrainedVectorParams> Beta_prm(bool check=true)const;
      virtual ParamVec t();
      virtual const ParamVec t()const;

      double a()const;
      double b()const;
      double d(uint m)const;
      const Vec & d()const;
      void set_a(double a);
      void set_b(double b);
      void set_d(const Vec &d);

      void fix_d0();
      void free_d0();
      bool is_d0_fixed()const;

      void initialize_params();
      void sync_params()const;

      virtual const Vec & beta()const;  // see note above for dimension
      void set_beta(const Vec &b);

      const Vec & fill_eta(const Vec &Theta)const;  // 0.. maxscore()
      const Mat & X(const Vec &Theta)const;
      const Mat & X(double theta)const;

      virtual double
      response_prob(Response r, const Vec &Theta, bool logsc)const;
      virtual double
      response_prob(uint r, const Vec &Theta, bool logsc)const;

      std::pair<double,double> theta_moments()const;
      // mean and variance of theta's for subjects that were assigned
      // this item

      virtual ostream &
      display_item_params(ostream &, bool decorate=true)const;

    private:
      // workspace for probability calculations
      mutable Vec b_, eta_;
      mutable Mat X_;
      bool d0_is_fixed;

      // pointers and flags for keeping track of alternate parameterizations
      mutable Ptr<ConstrainedVectorParams> beta_, d_prm;
      mutable Ptr<UnivParams> a_prm, b_prm;
      mutable bool beta_current, a_current, b_current, d_current;

      void impose_beta_constraint();
      void fill_beta(bool first_time=false)const;
      void fill_abd()const;
      void setup_X();
      void setup_beta();

      uint which_subscale_;

      // the observers watch a, b, and d for changes

      void observe_a()const{beta_current=false;}
      void observe_b()const{beta_current=false;}
      void observe_d()const{beta_current=false;}
      void observe_beta()const{a_current=b_current=d_current=false;}

      void set_abd_current()const;

      // to be called during construction:
      void setup();
      void setup_aliases();
      void set_observers();

      // helper for theta_moments
      void increment_theta_moments(Ptr<Subject>, double &m,
				   double &v, double &n)const;
    };
  }
}
#endif// BOOM_PARTIAL_CREDIT_MODEL_HPP
