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

#include <Models/IRT/DafePcr.hpp>
#include <cpputil/lse.hpp> // for lse and lse2
#include <cpputil/report_error.hpp>
#include <Models/IRT/PartialCreditModel.hpp>
#include <Models/IRT/Subject.hpp>
#include <Models/IRT/Item.hpp>
#include <distributions.hpp>
#include <boost/bind.hpp>
#include <stdexcept>

namespace BOOM{
  namespace IRT{

    typedef PartialCreditModel PCR;
    typedef DafePcrDataImputer IMP;

    IMP::DafePcrDataImputer(RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        Eta(0),
          mu(-0.577215664901533)
    {}

    void IMP::add_item(Ptr<PCR> mod){
      items.insert(mod);
      setup_latent_data(mod);
    }
    //------------------------------------------------------------
    void IMP::setup_latent_data(Ptr<PCR> mod){
      const SubjectSet &subjects(mod->subjects());
      for_each(subjects.begin(), subjects.end(),
           boost::bind(&IMP::setup_data_1, this, mod, _1));
    }
    //------------------------------------------------------------
    inline void mod_not_found(Ptr<PCR> mod, Ptr<Subject> s){
      ostringstream msg;
      msg << "item " << mod->id() << " not found  in subject "
      << s->id() << endl;
      report_error(msg.str());
    }
    //------------------------------------------------------------
    void IMP::setup_data_1(Ptr<PCR> mod, Ptr<Subject> s){
      //       cout << "setting up latent data for item " << mod->id()
      //        << " subject " << s->id() << endl;
      Response r = s->response(mod);
      if(!r) mod_not_found(mod, s);
      latent_data[r] = Vector(1+mod->maxscore());
    }
    //------------------------------------------------------------
    double IMP::logpri()const{return 0.0;}
    //------------------------------------------------------------
    Vector IMP::get_u(Response r, bool nag)const{
      std::map<Response, Vector>::const_iterator it = latent_data.find(r);
      if(it == latent_data.end()){
    if(nag){
      ostringstream msg;
      msg << "response not found in DafePcrDataImputer::get_u";
      report_error(msg.str());
    }
    return Vector();
      }
      const Vector &v(it->second);
      //      return  it->second;
      return  v;
    }
    //---------------- for debugging only ----------------------
    void IMP::set_u(Response r, const Vector &u){latent_data[r]=u; }
    //------------------------------------------------------------
    void IMP::draw(){
      std::for_each(items.begin(), items.end(),
            boost::bind(&IMP::draw_item_u, this, _1)); }
    //------------------------------------------------------------
    void IMP::draw_item_u(Ptr<PCR> mod){
      const SubjectSet & subjects(mod->subjects());
      for_each(subjects.begin(), subjects.end(),
           boost::bind(&IMP::draw_one, this, mod, _1));
    }
    //------------------------------------------------------------
    void IMP::draw_one(Ptr<PCR> mod, Ptr<Subject> s){
      Response r = s->response(mod);
      if(!r) mod_not_found(mod, s);
      Vector &u(latent_data[r]);
      Eta.resize(r->nlevels());
      const Vector &Eta(mod->fill_eta(s->Theta()));
      impute_u(u, Eta, r->value());
    }
    //------------------------------------------------------------
    void IMP::impute_u(Vector &u, const Vector &eta, uint y){
      double log_nc = lse(eta);
      double logzmin = rlexp(log_nc);
      uint M = u.size();
      for(uint m=0; m<M; ++m){
    if(m==y) u[m] = mu-logzmin;
    else u[m] = mu - lse2(logzmin, rlexp(eta[m]));}}

  } // namespace IRT
} // namespace BOOM
