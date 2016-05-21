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

#include <Models/Glm/VariableSelectionPrior.hpp>
#include <cpputil/math_utils.hpp>
#include <Models/BinomialModel.hpp>
#include <distributions.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  typedef VariableSelectionPrior VSP;

  namespace ModelSelection{

    Variable::Variable(uint pos, double prob, const string &name)
      : pos_(pos),
        mod_(new BinomialModel(1u, prob)),
        name_(name)
    {}

    Variable::Variable(const Variable & rhs)
      : RefCounted(rhs),
        pos_(rhs.pos_),
        mod_(rhs.mod_->clone()),
        name_(rhs.name_)
    {}

    Variable::~Variable() {}

    ostream & Variable::print(ostream &out)const{
      out << "Variable " << name_
          << " position " << pos_
          << " probability " << mod_->prob()
          << " ";
      return out;
    }

    ostream & operator<<(ostream &out, const Variable &v) {
      return v.print(out); }

    uint Variable::pos()const{return pos_;}
    double Variable::prob()const{return mod_->prob();}
    void Variable::set_prob(double prob) {
      mod_->set_prob(prob); }
    double Variable::logp(const Selector &inc)const{
      return mod_->pdf(inc[pos_], true);}
    Ptr<BinomialModel> Variable::model() {
      return mod_;}
    const Ptr<BinomialModel> Variable::model()const{
      return mod_;}

    const string & Variable::name()const{
      return name_;
    }

    //______________________________________________________________________
    typedef MainEffect ME;
    ME::MainEffect(uint pos, double prob, const string & name)
      : Variable(pos,prob, name)
    {}

    ME * ME::clone()const{return new ME(*this);}

    bool ME::observed()const{return true;}
    bool ME::parents_are_present(const Selector &)const{
      return true;}

    void ME::make_valid(Selector &inc)const{
      double p = prob();
      bool in = inc[pos()];
      if ((p>=1.0 && !in) || (p<=0.0 && in)) {
        inc.flip(pos());
      }
    }

    void ME::add_to(VSP & vsp)const{
      vsp.add_main_effect(pos(), prob(), name());}

    //______________________________________________________________________

    typedef MissingMainEffect MME;

    MME::MissingMainEffect(uint pos,
                           double prob,
                           uint obs_ind_pos,
                           const string &name)
      : MainEffect(pos,prob, name),
        obs_ind_pos_(obs_ind_pos)
      {}

    MME::MissingMainEffect(const MME & rhs)
      : MainEffect(rhs),
        obs_ind_pos_(rhs.obs_ind_pos_)
    {}

    MME * MME::clone()const{return new MME(*this);}

    double MME::logp(const Selector &inc)const{
      bool in = inc[pos()];
      bool oi_in = inc[obs_ind_pos_];
      if (oi_in) return Variable::logp(inc);
      return in ? BOOM::negative_infinity() : 0;
    }

    void MME::make_valid(Selector &inc)const{
      bool in = inc[pos()];
      double p = prob();
      if (p<=0.0 && in) {
        inc.drop(pos());
      }else if (p >= 1.0 && !in) {
        inc.add(pos());
        inc.add(obs_ind_pos_);
      }
    }

    bool MME::observed()const{return false;}

    bool MME::parents_are_present(const Selector &g)const{
      return g[obs_ind_pos_];}

    void MME::add_to(VSP & vsp)const{
      vsp.add_missing_main_effect(pos(), prob(), obs_ind_pos_, name()); }

    //______________________________________________________________________

    Interaction::Interaction(uint pos, double prob,
                             const std::vector<uint> &parents,
                             const string &name)
      : Variable(pos,prob, name),
        parent_pos_(parents)
    {}

    Interaction::Interaction(const Interaction &rhs)
      : Variable(rhs),
        parent_pos_(rhs.parent_pos_)
    {}

    Interaction * Interaction::clone()const{
      return new Interaction(*this);}

    uint Interaction::nparents()const{return parent_pos_.size();}

    double Interaction::logp(const Selector &inc)const{
      uint n = nparents();
      for (uint i = 0; i < n; ++i) {
        uint indx = parent_pos_[i];
        if (!inc[indx]) return BOOM::negative_infinity();
      }
      return Variable::logp(inc);
    }

    void Interaction::make_valid(Selector &g)const{
      double p = prob();
      bool in = g[pos()];
      if (p<=0.0 && in) {
        g.drop(pos());
      }else if (p>=1.0 && !in) {
        g.add(pos());
        for (int i = 0; i < parent_pos_.size(); ++i) {
          g.add(parent_pos_[i]);
        }
      }
    }

    bool Interaction::parents_are_present(const Selector &g)const{
      uint n = parent_pos_.size();
      for (uint i = 0; i < n; ++i) {
        if (!g[parent_pos_[i]]) return false;
      }
      return true;
    }

    void Interaction::add_to(VSP & vsp)const{
      vsp.add_interaction(pos(), prob(), parent_pos_, name());}
  }// closes namespace ModelSelection

  namespace ms=ModelSelection;
  //______________________________________________________________________

  VsSuf::VsSuf() {}

  VsSuf::VsSuf(const VsSuf & rhs)
    : Sufstat(rhs),
      SufTraits(rhs),
      vars_(rhs.vars_)
  {}

  VsSuf * VsSuf::clone()const{return new VsSuf(*this);}

  void VsSuf::add_var(VarPtr v) { vars_.push_back(v); }

  void VsSuf::clear() {
    uint n = vars_.size();
    for (uint i = 0; i < n; ++i)
      vars_[i]->model()->clear_suf();
  }

  void VsSuf::Update(const GlmCoefs &beta) {
    uint n = vars_.size();
    for (uint i = 0; i < n; ++i) {
      const Selector &g(beta.inc());
      if (vars_[i]->parents_are_present(g)) {
        bool y = g[i];
        vars_[i]->model()->suf()->update_raw(y);
      }
    }
  }

  void VsSuf::combine(Ptr<VsSuf>) {
    report_error("cannot combine VsSuf");
  }

  void VsSuf::combine(const VsSuf &) {
    report_error("cannot combine VsSuf");
  }

  VsSuf * VsSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this,s); }

  Vector VsSuf::vectorize(bool )const{
    report_error("cannot vectorize VsSuf");
    return Vector(1, 0.0);
  }

  Vector::const_iterator VsSuf::unvectorize(Vector::const_iterator &v,
                                            bool) {
    report_error("cannot unvectorize VsSuf");
    return v;
  }

  Vector::const_iterator VsSuf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & VsSuf::print(ostream &out)const{
    return out << "VsSuf is hard to print!";
  }


  //______________________________________________________________________


  VSP::VariableSelectionPrior()
    : DataPolicy(new VsSuf),
      pi_(new VectorParams(0))
  {}

  VSP::VariableSelectionPrior(uint n, double pi)
    : DataPolicy(new VsSuf),
      pi_(new VectorParams(0))
  {
    for (uint i = 0; i < n; ++i) {
      add_main_effect(i, pi);
    }
  }

  VSP::VariableSelectionPrior(const Vector &pi)
    : DataPolicy(new VsSuf),
      pi_(new VectorParams(0))
  {
    uint n = pi.size();
    for (uint i = 0; i < n; ++i) {
      add_main_effect(i, pi[i]);
    }
  }

  VSP::VariableSelectionPrior(const VSP & rhs)
    : Model(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      pi_(new VectorParams(rhs.pi_->size()))
  {
    uint n = rhs.vars_.size();
    for (uint i = 0; i < n; ++i) {
      rhs.vars_[i]->add_to(*this);
    }
  }

  VSP * VSP::clone()const{ return new VSP(*this);}

  // double VSP::loglike()const{
  //   double ans = 0;
  //   uint n = vars_.size();
  //   for (uint i = 0; i < n; ++i) {
  //     ans += vars_[i]->model()->loglike();
  //   }
  //   return ans;
  // }

  void VSP::check_size_eq(uint n, const string &fun)const{
    if (vars_.size()==n) return;
    ostringstream err;
    err << "error in VSP::" << fun << endl
        << "you passed a vector of size " << n
        << " but there are " << vars_.size()
        << " variables." <<endl;
    report_error(err.str());
  }

  void VSP::check_size_gt(uint n, const string &fun)const{
    if (vars_.size()>n) return;
    ostringstream err;
    err << "error in VSP::" << fun << endl
        << "you tried to access a variable at position " << n
        << ", but there are only " << vars_.size()
        << " variables." << endl;
    report_error(err.str());
  }
  void VSP::set_prob(double p, uint i) {
    check_size_gt(i, "set_prob");
    vars_[i]->set_prob(p);
  }

  void VSP::set_probs(const Vector & pi) {
    uint n = pi.size();
    check_size_eq(n, "set_probs");
    for (uint i = 0; i < n; ++i) vars_[i]->set_prob(pi[i]);
  }

  Vector VSP::prior_inclusion_probabilities() const {
    Vector ans(potential_nvars());
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] = prob(i);
    }
    return ans;
  }

  double VSP::prob(uint i)const{
    check_size_gt(i, "prob");
    return vars_[i]->prob();
  }

  void VSP::fill_pi()const{
    uint n = vars_.size();
    Vector tmp(n);
    for (uint i = 0; i < n; ++i) tmp[i] = vars_[i]->prob();
    pi_->set(tmp);
  }

  ParamVector VSP::t() {
    fill_pi();
    return ParamVector(1,pi_); }

  const ParamVector VSP::t()const{
    fill_pi();
    return ParamVector(1,pi_);}

  void VSP::unvectorize_params(const Vector &v, bool) {
    uint n =v.size();
    check_size_eq(n, "unvectorize_params");
    for (uint i = 0; i < n; ++i) {
      double p = v[i];
      vars_[i]->model()->set_prob(p);
    }
  }

  void VSP::mle() {
    uint n = vars_.size();
    for (uint i = 0; i < n; ++i) vars_[i]->model()->mle();
  }

  double VSP::pdf(Ptr<Data> dp, bool logscale)const{
    Ptr<GlmCoefs> d(DAT(dp));
    double ans = logp(d->inc());
    return logscale ? ans : exp(ans);
  }

  VSP::VarPtr VSP::variable(uint i) { return vars_[i]; }
  const VSP::VarPtr VSP::variable(uint i)const{
    return vars_[i]; }

  namespace {
    inline void draw(Ptr<ModelSelection::Variable> v, Selector &g, RNG &rng) {
      double u =runif_mt(rng, 0,1);
      uint pos = v->pos();
      if (u < v->prob()) g.add(pos);
    }
  }

  Selector VSP::simulate(RNG &rng)const{
    uint n = potential_nvars();
    Selector ans(n,false);
    // simulate main_effects
    uint nme = observed_main_effects_.size();
    for (uint i = 0; i < nme; ++i)
      draw(observed_main_effects_[i], ans, rng);

    // simulate missing main effects
    uint nmis = missing_main_effects_.size();
    for (uint i = 0; i < nmis; ++i) {
      Ptr<MissingMainEffect> v = missing_main_effects_[i];
      if (v->parents_are_present(ans)) draw(v,ans, rng);
    }

    uint nint = interactions_.size();
    for (uint i = 0; i <  nint; ++i) {
      Ptr<Interaction> v = interactions_[i];
      if (v->parents_are_present(ans)) draw(v, ans, rng);
    }
    return ans;
  }

  uint VSP::potential_nvars()const{ return vars_.size();}

  double VSP::logp(const Selector &inc)const{
    const double neg_inf = BOOM::negative_infinity();
    //    if (inc.nvars()==0) return neg_inf;
    uint n = vars_.size();
    double ans=0;
    for (uint i = 0; i < n; ++i) {
      ans += vars_[i]->logp(inc);
      if (ans<=neg_inf) {
        return ans;
      }
    }
    return ans;
  }

  void VSP::make_valid(Selector &inc)const{
    int n = vars_.size();
    for (int i = 0; i < n; ++i) {
      vars_[i]->make_valid(inc);
    }
  }

  void VSP::add_main_effect(uint pos, double prob, const string &name) {
    NEW(MainEffect, me)(pos,prob, name);
    observed_main_effects_.push_back(me);
    VarPtr v(me);
    vars_.push_back(v);
    suf()->add_var(v);
  }

  void VSP::add_missing_main_effect(uint pos,
                                    double prob,
                                    uint oi_pos,
                                    const string &name) {
    NEW(MissingMainEffect, mme)(pos, prob, oi_pos, name);
    suf()->add_var(mme);
    vars_.push_back(VarPtr(mme));
    missing_main_effects_.push_back(mme);
  }

  void VSP::add_interaction(uint pos,
                            double prob,
                            const std::vector<uint> & parents,
                            const string &name) {
    NEW(Interaction, inter)(pos, prob, parents, name);
    VarPtr v(inter);
    vars_.push_back(v);
    suf()->add_var(v);
    interactions_.push_back(inter);
  }

  ostream & VSP::print(ostream &out)const{
    uint nv = vars_.size();
    for (uint i = 0; i < nv; ++i) {
      out << *(vars_[i]) << endl;
    }
    return out;
  }

  ostream & operator<<(ostream &out, const VSP &vsp) {
    return vsp.print(out);}

}
