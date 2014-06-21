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

#ifndef BOOM_MARKOV_MODEL_HPP
#define BOOM_MARKOV_MODEL_HPP

#include <vector>
#include <BOOM.hpp>

#include <Models/ModelTypes.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/CategoricalData.hpp>

#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>

#include <Models/TimeSeries/MarkovLink.hpp>
#include <Models/TimeSeries/TimeSeries.hpp>
#include <Models/TimeSeries/TimeSeriesSufstatDataPolicy.hpp>

#include <Models/Sufstat.hpp>
#include <LinAlg/Types.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>


//=====================================================================
namespace BOOM{

  //====================================================================
  class MarkovData :  public CategoricalData{
    MarkovLink<MarkovData> links;
  public:
    //------- onstructors, destructors, copy, = ==
    MarkovData(uint val, uint Nlevels);
    MarkovData(uint val, pKey labs);
    MarkovData(const string &lab, pKey labs, bool grow=false);

    MarkovData(uint val, Ptr<MarkovData> last);
    MarkovData(const string &lab, Ptr<MarkovData> last, bool grow=false);

    MarkovData(const MarkovData &, bool copy_links=false);

    virtual MarkovData * create()const;  // does not copy links
    virtual MarkovData * clone()const;   // copies links

    MarkovData * prev()const;
    MarkovData * next()const;
    void unset_prev();
    void unset_next();
    void clear_links();
    void set_prev(Ptr<MarkovData > p);
    void set_next(Ptr<MarkovData > n);

    ostream & display(ostream & )const;
    //    istream & read(istream &in);
  };
  //=====================================================================
  typedef TimeSeries<MarkovData> MarkovDataSeries ;
  Ptr<MarkovDataSeries> make_markov_data(const std::vector<uint> &raw_data,
					 bool full_range=true);
  Ptr<MarkovDataSeries> make_markov_data(const std::vector<string> & raw_data);
  Ptr<MarkovDataSeries> make_markov_data(const std::vector<string> & raw_data,
					 const std::vector<string> &order);

  //=====================================================================
  const bool debug_markov_update_suf(false);
  class MarkovSuf
    : public TimeSeriesSufstatDetails<MarkovData, MarkovDataSeries>{
  public:

    MarkovSuf(uint S);
    MarkovSuf(const MarkovSuf &sf);
    MarkovSuf *clone() const;

    uint state_space_size()const{ return trans().nrow() ;}
    void resize(uint p);
    void clear(){ trans_=0.0; init_=0.0; }
    void Update(const MarkovData &);
    void add_mixture_data(Ptr<MarkovData>, double prob);
    void add_transition_distribution(const Mat &P);
    void add_initial_distribution(const Vec &pi);
    void add_transition(uint from, uint to);
    void add_initial_value(uint val);
    const Mat & trans()const{return trans_;}
    const Vec & init()const{return init_;}
    std::ostream &print(std::ostream &)const;
    void combine(Ptr<MarkovSuf>);
    void combine(const MarkovSuf &);
    MarkovSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
  private:
    Mat trans_; // transition counts
    Vec init_; // initial count, typically one 1 and rest 0's
  };
  //=====================================================================
  std::ostream & operator<<(std::ostream &out, Ptr<MarkovSuf> sf);
  //=====================================================================

  //------ observer classes ------------------
  class MatrixRowsObserver{
  public:
    typedef std::vector<Ptr<VectorParams> > Rows;
    MatrixRowsObserver(Rows&);
    void operator()(const Mat &);
  private:
    Rows  & rows;
  };

  class StationaryDistObserver{
  public:
    StationaryDistObserver(Ptr<VectorParams>);
    void operator()(const Mat &);
  private:
    Ptr<VectorParams> stat;
  };

  class RowObserver{
  public:
    RowObserver(Ptr<MatrixParams> M, uint I);
    void operator()(const Vec &v);
  private:
    Ptr<MatrixParams> mp;
    Mat m;
    uint i;
  };

  //======================================================================

  class TransitionProbabilityMatrix
     : public MatrixParams{
  public:
    TransitionProbabilityMatrix(uint S);
    TransitionProbabilityMatrix(const Mat &);
    TransitionProbabilityMatrix(const TransitionProbabilityMatrix &);
    TransitionProbabilityMatrix * clone()const;

    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                            bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v, bool minimal=true);
    virtual void set(const Mat &m, bool signal=true);

    void add_observer(Ptr<VectorParams>)const;
    void delete_observer(Ptr<VectorParams>)const;
  private:
    typedef std::set<Ptr<VectorParams> > ObsSet;
    mutable ObsSet observers;
    void notify()const;
  };

  //======================================================================
  class ProductDirichletModel;
  class DirichletModel;
  class MarkovConjSampler;

  class MarkovModel
    : public ParamPolicy_2<TransitionProbabilityMatrix,VectorParams>,
      public TimeSeriesSufstatDataPolicy<MarkovData, MarkovDataSeries,
                                         MarkovSuf>,
      public ConjugatePriorPolicy<MarkovConjSampler>,
      public LoglikeModel,
      public EmMixtureComponent
  {

  public:
    typedef MarkovData DataPointType;
    typedef MarkovDataSeries DataSeriesType;
    typedef TransitionProbabilityMatrix TPM;

    MarkovModel(uint S);
    MarkovModel(const Mat &Q);
    MarkovModel(const Mat & Q, const Vec & pi0);
    MarkovModel(const std::vector<uint> & );
    MarkovModel(const std::vector<string> & );

    MarkovModel(const MarkovModel &rhs);
    virtual MarkovModel * clone()const;

    void fix_pi0(const Vec &Pi0);
    void fix_pi0_stationary();
    void fix_pi0_uniform();
    void free_pi0();
    bool pi0_fixed()const;

    double pdf(Ptr<Data> dp, bool logscale) const;
    double pdf(const Data * dp, bool logscale) const;
    double pdf(Ptr<DataPointType> dp, bool logscale) const;
    double pdf(Ptr<DataSeriesType> dp, bool logscale) const;
    double pdf(const DataPointType &dat, bool logscale) const;
    double pdf(const DataSeriesType &dat, bool logscale) const;

    virtual void add_mixture_data(Ptr<Data>, double prob);

    uint state_space_size()const;

    Ptr<TPM> Q_prm();
    const Ptr<TPM> Q_prm()const;
    virtual const Mat &Q()const;
    virtual void set_Q(const Mat &Q)const;
    double Q(uint, uint)const;

    Ptr<VectorParams> Pi0_prm();
    const Ptr<VectorParams> Pi0_prm()const;
    virtual const Vec & pi0()const;
    void set_pi0(const Vec &pi0);
    double pi0(int)const;

    void mle();
    virtual void find_posterior_mode();
    void set_conjugate_prior(Ptr<ProductDirichletModel>);
    void set_conjugate_prior(Ptr<ProductDirichletModel>, Ptr<DirichletModel>);
    void set_conjugate_prior(Ptr<MarkovConjSampler>);

    double loglike()const;
    Vec stat_dist()const;

  protected:
    virtual void resize(uint S);
  private:
    Ptr<MarkovData> dpp;  // data point prototype
    enum Pi0Status{Free, Uniform, Stationary, Known};
    Pi0Status pi0_status;
  };


} // ends namespace BOOM

#endif // MARKOV_MODEL_H
