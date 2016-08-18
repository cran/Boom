/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_BETA_BINOMIAL_MODEL_HPP
#define BOOM_BETA_BINOMIAL_MODEL_HPP

#include <cstdint>
#include <Models/DataTypes.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{

  class BinomialData : public Data{
   public:
    BinomialData(int64_t n = 0, int64_t y = 0);
    BinomialData(const BinomialData &rhs);
    BinomialData * clone()const override;
    BinomialData & operator=(const BinomialData &rhs);

    virtual uint size(bool minimal = true)const;
    ostream &display(ostream &)const override;

    int64_t trials()const;
    int64_t n()const;
    void set_n(int64_t trials);

    int64_t y()const;
    int64_t successes()const;
    void set_y(int64_t successes);
   private:
    int64_t trials_;
    int64_t successes_;

    void check_size(int64_t n, int64_t y)const;
  };

  // BetaBinomialModel describes a setting were binomial data occurs
  // within groups.  Each group has its own binomial success
  // probability drawn from a beta(a, b) distribution.  If the group
  // size is 1 then this is simply the BetaBinomial distribution.

  class BetaBinomialModel
    : public ParamPolicy_2<UnivParams, UnivParams>,
      public IID_DataPolicy<BinomialData>,
      public PriorPolicy,
      public NumOptModel
  {
   public:
    BetaBinomialModel(double a, double b);

    // Using this constructor will initialize the model with one of
    // three sets of parameters.
    // a) If a call to mle() succeeds then the parameters will be set
    //    using maximum likelihood estimates.
    // b) If the call to mle() fails then the parameters will be set
    //    using a call to method_of_moments().
    // c) If the call to method_of_moments() fails then a and b will
    //    both be set to 1.0.
    // Args:
    //   trials:  The number of trials observed, per group.
    //   successes: The number of successes observed per group (must
    //     be <= the number of trials.)
    BetaBinomialModel(const BOOM::Vector &trials,
                      const BOOM::Vector &successes);
    BetaBinomialModel(const BetaBinomialModel &rhs);
    BetaBinomialModel *clone()const override;

    void clear_data() override;
    void add_data(Ptr<Data> data) override;
    void add_data(Ptr<BinomialData> data) override;

    // The likelihood contribution for observation i is
    // int Pr(y_i | theta_i, n_i) p(theta_i) dtheta_i
    virtual double loglike()const;
    double loglike(const Vector &ab)const override;
    double loglike(double a, double b)const;
    double Loglike(const Vector &ab, Vector &g, Matrix &H, uint nd)const override;
    double logp(int64_t n, int64_t y, double a, double b)const;

    // Args:
    //   n:  The number of trials for a particular observation.  All trials will
    //     have the same success probability.
    // Returns:
    //   The number of successes for the observation in question.
    int64_t sim(int64_t n)const;

    Ptr<UnivParams> SuccessPrm();
    Ptr<UnivParams> FailurePrm();
    const Ptr<UnivParams> SuccessPrm()const;
    const Ptr<UnivParams> FailurePrm()const;
    double a()const;
    void set_a(double a);
    double b()const;
    void set_b(double b);

    double prior_mean()const;             // a / a+b
    void set_prior_mean(double prob);

    double prior_sample_size()const;      // a+b
    void set_prior_sample_size(double sample_size);

    // Set a/(a+b) and a+b using a very rough method of moments
    // estimator.  The estimator can fail if either the sample mean or
    // the sample variance is zero, in which case this function will
    // exit without changing the model.
    void method_of_moments();

    // Print a summary of the model on the stream 'out', and return
    // 'out'.
    std::ostream & print_model_summary(std::ostream &out) const;

   private:
    void check_positive(double arg, const char *function_name)const;
    void check_probability(double arg, const char *function_name)const;

    // Stores sum of lgammafn(n+1) - lgammafn(y+1) - lgammafn(n-y+1) since the
    // trials and rewards do not change over the lifetime of this object.
    double lgamma_n_y_;
  };

}  // namespace BOOM

#endif //  BOOM_BETA_BINOMIAL_MODEL_HPP
