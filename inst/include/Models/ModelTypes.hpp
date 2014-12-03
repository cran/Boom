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
#ifndef BOOM_MODEL_TYPES_HPP
#define BOOM_MODEL_TYPES_HPP

#include <BOOM.hpp>
#include <Models/ParamTypes.hpp>
#include <LinAlg/Types.hpp>
#include <boost/shared_ptr.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  class PosteriorSampler;
  class Params;
  class Data;

  // A Model is the basic unit of operation in statistical learning.
  // In BOOM, each Model manages Params, Data, and learning methods.
  // a Model should also inherit from a ParamPolicy, a DataPolicy, and
  // a PriorPolicy.
  //
  // If the recommended set of policies are used, to inherit from
  // Model a class must provide:
  //   * clone()   (covariant return)
  //
  // Any class inheriting from model should do so virtually, because
  // Model contains a reference count that should not be duplicated.
  class Model : private RefCounted{
  public:
    friend void intrusive_ptr_add_ref(Model *d){d->up_count();}
    friend void intrusive_ptr_release(Model *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

    //------ constructors, destructors, operator=/== -----------
    Model();
    Model(const Model &rhs);        // ref count is not copied
    virtual Model * clone()const=0;

    // the result of clone() should have identical parameters in
    // distinct memory.  It should not have any data assigned.  Nor
    // should it include the same priors and sampling methods

    virtual ~Model(){}

    //----------- parameter interface  ---------------------
    virtual ParamVec t()=0;              // implemented in ParmPolicy
    virtual const ParamVec t()const=0;

    virtual Vec vectorize_params(bool minimal=true)const;
    virtual void unvectorize_params(const Vec &v, bool minimal=true);

    //------------ functions implemented in DataPolicy -----

    // add_data adds 'dp' to the set of Data objects managed by the
    // model.  It is assumed that dp points to a Data object of the
    // type produced by the concrete model.  The Data type is made
    // concrete in the model's DataPolicy.
    virtual void add_data(Ptr<Data> dp)=0;    //

    // Discard all the data that has been added using add_data().
    virtual void clear_data()=0;

    // Combine the data managed by other_model with the data managed
    // by *this.  If just_suf is true and the model has sufficient
    // statistics, the actual data from 'other_model' is not copied.
    virtual void combine_data(const Model & other_model, bool just_suf=true)=0;

    //------------ functions over-ridden in PriorPolicy ----
    virtual void sample_posterior()=0;
    virtual double logpri()const=0;      // evaluates current params
    virtual void set_method(Ptr<PosteriorSampler>)=0;
  };

  //============= mix-in classes =========================================

  // The model has parameters that can be estimated by maximum likelihood.
  class MLE_Model : virtual public Model {
  public:
    MLE_Model() : status_(NOT_CALLED) {}
    // Set the paramters to their maximum likelihood estimates.
    virtual void mle()=0;
    virtual void initialize_params();
    virtual MLE_Model *clone()const=0;
    enum MleStatus {
      NOT_CALLED = -1,
      FAILURE = 0,
      SUCCESS = 1
    };
    MleStatus mle_status() const {return status_;}
    const std::string & mle_error_message() const {
      return error_message_;
    }
    bool mle_success() const {
      return status_ == SUCCESS;
    }

   private:
    MleStatus status_;
    std::string error_message_;

   protected:
    void set_status(MleStatus status, const string &error_message) {
      status_ = status;
      error_message_ = error_message;
    }
  };

  class LoglikeModel : public MLE_Model {
  public:
    virtual double loglike(const Vector &theta)const=0;
    virtual LoglikeModel * clone()const=0;
    virtual void mle();
  };

  class dLoglikeModel : public LoglikeModel{
  public:
    virtual double dloglike(const Vector &x, Vec &g)const=0;
    virtual void mle();
    virtual dLoglikeModel *clone()const=0;
  };

  class d2LoglikeModel : public dLoglikeModel{
  public:
    virtual double d2loglike(const Vector &x, Vec &g, Mat &H)const=0;
    virtual void mle();
    virtual double mle_result(Vec &gradient, Mat &hessian);
    virtual d2LoglikeModel *clone()const=0;
  };

  class NumOptModel : public d2LoglikeModel{
  public:
    virtual double Loglike(const Vector &x, Vec &g, Mat &H, uint nd)const=0;
    virtual double loglike(const Vector &x)const{
      Vec g;
      Mat h;
      return Loglike(x, g, h, 0);
    }
    virtual double dloglike(const Vector &x, Vec &g)const{
      Mat h;
      return Loglike(x, g, h, 1);
    }
    virtual double d2loglike(const Vector &x, Vec &g, Mat &h)const{
      return Loglike(x, g, h, 2);
    }
    virtual NumOptModel * clone()const=0;
  };
  //======================================================================
  class LatentVariableModel {
  public:
    virtual void impute_latent_data(RNG &rng)=0;
    virtual LatentVariableModel * clone()const=0;
  };
  //======================================================================
  class CorrModel : virtual public Model {
  public:
    virtual CorrModel * clone()const=0;
    virtual double logp(const Corr &)const=0;
  };
  //======================================================================
  class MixtureComponent : virtual public Model {
   public:
    virtual double pdf(const Data *, bool logscale)const=0;
    virtual MixtureComponent * clone()const=0;
  };

}  // namespace BOOM
#endif // BOOM_MODEL_TYPES_HPP
