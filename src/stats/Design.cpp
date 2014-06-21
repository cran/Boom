/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#include <stats/Design.hpp>
#include <algorithm>
#include <cpputil/DefaultVnames.hpp>
#include <BOOM.hpp>
#include <iostream>
#include <iomanip>

namespace BOOM{
  DesignMatrix::DesignMatrix(const Mat &X)
    : Mat(X)
  {
    vnames_ = default_vnames(X.ncol());
  }

  DesignMatrix::DesignMatrix
  (const Mat &X, const std::vector<string> &vnames)
    : Mat(X),
      vnames_(vnames)
  { }

  DesignMatrix::DesignMatrix
  (const Mat &X, const std::vector<string> &vnames,
   const std::vector<string> &baseline_names)
    : Mat(X),
      vnames_(vnames),
      baseline_names_(baseline_names)
  {}

  DesignMatrix::DesignMatrix(const DesignMatrix &rhs)
    : Mat(rhs),
      vnames_(rhs.vnames_),
      baseline_names_(rhs.baseline_names_)
  {}



  //-----------------------------------------------------------------
  ostream & display(ostream & out, const DesignMatrix &m,
                    int prec, uint from, uint to){
    using std::setw;
    using std::setprecision;

    out << setprecision(prec);
    const std::vector<string> &vn(m.vnames());
    uint nvars = vn.size();
    std::vector<uint> fw(nvars);

    uint padding = 2;
    for(uint i=0; i<nvars; ++i){
      fw[i] = std::max<uint>(8u, vn[i].size()+padding);
    }

    for(uint i=0; i<m.vnames().size(); ++i)
      out << setw(fw[i]) << vn[i] ;
    out << endl;

    if(to>m.nrow()) to = m.nrow();
    for(uint i=from; i<to; ++i){
      for(uint j =0; j<m.ncol(); ++j){
        out << setw(fw[j]) << m(i,j);
      }
      out << endl;
    }
    return out;
  }

  //-----------------------------------------------------------------
  ostream & operator<<(ostream &out, const DesignMatrix &dm){
    return display(out, dm);  }
  //-----------------------------------------------------------------


  typedef std::map<string, std::vector<string> > LevelNames;
  DesignMatrix generate_design_matrix(const LevelNames & level_names, int order){
    ExperimentStructure xp(level_names);
    RowBuilder builder(xp, order);
    Configuration config(xp.nlevels());
    int nr = xp.nconfigurations();
    int nc = builder.dim();
    Mat X(nr, nc);
    int i = 0;
    while(!config.done()){
      X.row(i++) = builder.build_row(config);
      config.next();
    }
    std::vector<string> vnames = builder.variable_names();
    DesignMatrix ans(X, vnames, xp.baseline_levels());
    return ans;
  }

  ExperimentStructure::ExperimentStructure(const LevelNames &level_names){
    for(LevelNames::const_iterator it = level_names.begin();
        it != level_names.end(); ++it){
      factor_names_.push_back(it->first);
      level_names_.push_back(it->second);
      nlevels_.push_back(it->second.size());
    }
  }

  ExperimentStructure::ExperimentStructure(const std::vector<int> & nlevels)
      : factor_names_(generate_names(nlevels.size())),
        nlevels_(nlevels)
  {
    int nfactors = nlevels.size();
    level_names_.reserve(nfactors);
    for(int i = 0; i < nfactors; ++i){
      level_names_.push_back(generate_names(nlevels_[i]));
    }
  }

  std::vector<string> ExperimentStructure::generate_names(int n)const{
    std::vector<string> ans;
    ans.reserve(n);
    for(int i = 0; i < n; ++i){
      ostringstream name;
      name << i;
      ans.push_back(name.str());
    }
    return ans;
  }

  int ExperimentStructure::nfactors()const{
    return factor_names_.size();
  }

  int ExperimentStructure::nlevels(int factor)const{
    return level_names_[factor].size();
  }

  const std::vector<int> & ExperimentStructure::nlevels()const{
    return nlevels_;
  }

  int ExperimentStructure::nconfigurations()const{
    int ans = 1;
    for(int i = 0; i < nfactors(); ++i) ans *= nlevels(i);
    return ans;
  }

  const string & ExperimentStructure::level_name(int factor, int level) const{
    return level_names_[factor][level]; }


  std::vector<string> ExperimentStructure::baseline_levels() const{
    std::vector<string> ans;
    ans.reserve(nfactors());
    for(int i = 0; i < nfactors(); ++i){
      ans.push_back(level_names_[i].front());
    }
    return ans;
  }
  //----------------------------------------------------------------------
  Configuration::Configuration(const std::vector<int> &nlevels)
    : nlevels_(nlevels),
      levels_(nlevels.size(), 0)
 {}

  Configuration::Configuration(const std::vector<int> &nlevels,
                               const std::vector<int> &levels)
    : nlevels_(nlevels),
      levels_(levels)
 {}

  void Configuration::next(){
    if(done()) return;
    int pos = levels_.size() - 1;
    while (pos >= 0){
      ++levels_[pos];
      if(levels_[pos] < nlevels_[pos]) return;
      levels_[pos] = 0;
      --pos;
    }
    // if you've gotten here then all levels have seen their maximum value
    levels_.assign(levels_.size(), -1);
  }

  bool Configuration::done()const{return levels_[0] == -1;}

  int Configuration::level(int factor)const{
    return levels_[factor];
  }

  const std::vector<int> & Configuration::levels()const{
    return levels_;
  }

  bool Configuration::operator==(const Configuration &rhs)const{
    return (levels_ == rhs.levels_) && (nlevels_ == rhs.nlevels_);
  }

  bool Configuration::operator!=(const Configuration &rhs)const{
    return !(*this == rhs);
  }

  ostream & Configuration::print(ostream &out)const{
    int n = levels_.size();
    if(n==0) return out;
    out << levels_[0];
    for(int i = 1; i < n; ++i){
      out << " " << levels_[i];
    }
    return out;
  }

  //----------------------------------------------------------------------
  FactorDummy::FactorDummy(int factor, int level, const string &name)
    : factor_(factor),
      level_(level),
      name_(name)
 {}

  bool FactorDummy::eval(const Configuration & config)const{
    return config.level(factor_) == level_;
  }

  bool FactorDummy::eval(const std::vector<int> &levels)const{
    return levels[factor_] == level_;
  }

  const string & FactorDummy::name()const{ return name_; }

  bool FactorDummy::operator==(const FactorDummy &rhs)const{
    return (factor_ == rhs.factor_) && (level_ == rhs.level_);
  }

  bool FactorDummy::operator<(const FactorDummy &rhs)const{
    if(factor_ < rhs.factor_) return true;
    if(factor_ > rhs.factor_) return false;
    return level_ < rhs.level_;
  }

  int FactorDummy::factor()const{return factor_;}
  //----------------------------------------------------------------------
  Effect::Effect(){}
  Effect::Effect(const FactorDummy &factor){
    add_factor(factor);
  }
  Effect::Effect(const Effect &first, const Effect &second){
    std::copy(first.factors_.begin(), first.factors_.end(),
              back_inserter(factors_));
    std::copy(second.factors_.begin(), second.factors_.end(),
              back_inserter(factors_));
    std::sort(factors_.begin(), factors_.end());
    std::unique(factors_.begin(), factors_.end());
  }

  int Effect::order()const{
    return factors_.size();
  }

  void Effect::add_factor(const FactorDummy &factor){
    if(!has_factor(factor)) factors_.push_back(factor);
    std::sort(factors_.begin(), factors_.end());
  }

  void Effect::add_effect(const Effect &effect){
    int nef = effect.factors_.size();
    for(int i = 0; i < nef; ++i){
      add_factor(effect.factors_[i]);}}

  bool Effect::eval(const Configuration &config)const{
    for(int i = 0; i < factors_.size(); ++i){
      if(!factors_[i].eval(config)) return false;
    }
    return true;
  }

  bool Effect::eval(const std::vector<int> &levels)const{
    for(int i = 0; i < factors_.size(); ++i){
      if(!factors_[i].eval(levels)) return false;
    }
    return true;
  }

  string Effect::name()const{
    int nterms = factors_.size();
    if(nterms == 0) return "Intercept";
    string ans = factors_[0].name();
    for(int i = 1; i < nterms; ++i){
      ans += ":";
      ans += factors_[i].name();
    }
    return ans;
  }

  bool Effect::operator==(const Effect &rhs)const{
    return factors_ == rhs.factors_;
  }

  bool Effect::operator<(const Effect &rhs)const{
    return factors_ < rhs.factors_; }

  // returns true if the effect already has a factor from the same
  // factor family as the factor dummy
  bool Effect::has_factor(const FactorDummy &factor)const{
    int factor_value = factor.factor();
    for(int i = 0; i < factors_.size(); ++i){
      if(factors_[i].factor() == factor_value) return true;
    }
    return false;
  }

  const FactorDummy & Effect::factor(int factor) const{
    return factors_[factor];}

  //----------------------------------------------------------------------
  RowBuilder::RowBuilder(){}
  RowBuilder::RowBuilder(const ExperimentStructure &xp, int order){
    setup(xp, order);
  }

  RowBuilder::RowBuilder(const std::vector<int> &nlevels, int order){
    ExperimentStructure xp(nlevels);
    setup(xp, order);
  }

  RowBuilder::RowBuilder(const std::map<string, std::vector<string> > &level_names,
                         int order){
    ExperimentStructure xp(level_names);
    setup(xp, order);
  }

  void RowBuilder::setup(const ExperimentStructure &xp, int interaction_order){
    Effect intercept;
    add_effect(intercept);
    if(interaction_order == 0) return;

    std::vector<Effect> main_effects;
    for(int factor = 0; factor < xp.nfactors(); ++factor){
      for(int level = 1; level < xp.nlevels(factor); ++level){
        FactorDummy dummy(factor, level, xp.level_name(factor, level));
        Effect main_effect(dummy);
        effects_.push_back(main_effect);
        main_effects.push_back(main_effect);}}

    int nmain = main_effects.size();
    std::vector<Effect> last_effects = main_effects;

    for(int order = 2; order <= interaction_order; ++order){
      std::vector<Effect> current_effects;
      for(int i = 0; i < nmain; ++i){
        Effect main_effect = main_effects[i];
        for(int j = 0; j < last_effects.size(); ++j){
          Effect lower_order = last_effects[j];
          if(lower_order.has_factor(main_effect.factor(0))){
            continue;
          }
          Effect interaction(main_effect, lower_order);
          if(has_effect(interaction)){
            continue;
          }
          current_effects.push_back(interaction);
          effects_.push_back(interaction);}}
      last_effects = current_effects;}
  }

  bool RowBuilder::has_effect(const Effect &effect)const{
    return std::find(effects_.begin(), effects_.end(), effect)
        != effects_.end();
  }

  void RowBuilder::add_effect(const Effect &e){
    effects_.push_back(e);
  }

  int RowBuilder::number_of_main_effects()const{
    int ans = 0;
    for(int i = 0; i < effects_.size(); ++i){
      ans += (effects_[i].order() == 1); // excludes intercept, with order()==0
    }
    return ans;
  }

  const Effect & RowBuilder::effect(int i)const{
    return effects_[i];
  }

  std::vector<double> RowBuilder::build_row(const Configuration &config) const{
    int neffects = effects_.size();
    std::vector<double> ans(neffects);
    for(int i = 0; i < neffects; ++i) ans[i] = effects_[i].eval(config);
    return ans;
  }

  std::vector<double> RowBuilder::build_row(const std::vector<int> &levels) const{
    int neffects = effects_.size();
    std::vector<double> ans(neffects);
    for(int i = 0; i < neffects; ++i) ans[i] = effects_[i].eval(levels);
    return ans;
  }

  int RowBuilder::dim()const{ return effects_.size(); }

  std::vector<string> RowBuilder::variable_names()const{
    std::vector<string> ans;
    int neffects = effects_.size();
    ans.reserve(neffects);
    for(int i = 0; i < neffects; ++i){ ans.push_back(effects_[i].name()); }
    return ans;
  }
}
