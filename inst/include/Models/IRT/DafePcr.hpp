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
#ifndef BOOM_DAFE_PCR_HPP
#define BOOM_DAFE_PCR_HPP

#include <Models/VectorModel.hpp>
#include <Models/IRT/IRT.hpp>
#include <Models/IRT/PartialCreditModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Samplers/MetropolisHastings.hpp>
#include <map>

namespace BOOM{
  class MvnModel;
  class MvtModel;
  class IndepMH;

  namespace IRT{
    class PartialCreditModel;

    class DafePcrDataImputer : public PosteriorSampler{
    public:
      typedef PartialCreditModel PCR;
      DafePcrDataImputer();
      void add_item(Ptr<PCR>);
      void draw();
      double logpri()const;
      Vec get_u(Response r, bool nag=true)const;

      // ---- for debugging purposes only -----
      void set_u(Response r, const Vec &u);
      //---------------------------------------
    private:
      // this object stores internal data from partial credit models
      std::set<Ptr<PCR> > items;
      std::map<Response, Vec> latent_data;  // "u" from scott 2006
      Vec Eta;                    // workspace
      const double mu;            // -1* Euler's constant

      //--- internal helper functions--
      void setup_latent_data(Ptr<PCR>);
      void setup_data_1(Ptr<PCR>, Ptr<Subject>);
      void impute_u(Vec &u, const Vec & Eta, uint y);
      void draw_item_u(Ptr<PCR>);
      void draw_one(Ptr<PCR>, Ptr<Subject>);
    };

    //============================================================
    class DafePcrItemSampler : public PosteriorSampler{
    public:
      DafePcrItemSampler(Ptr<PartialCreditModel>,
			 Ptr<DafePcrDataImputer>,
			 Ptr<MvnModel> Prior,
			 double Tdf);
      void draw();
      double logpri()const;
    private:
      Ptr<PartialCreditModel> mod;
      Ptr<MvnModel> prior;
      Ptr<DafePcrDataImputer> imp;
      Ptr<MetropolisHastings> sampler;
      Ptr<MvtIndepProposal> prop;

      const double sigsq;  //  = pi^2/6 = 1.64493406684
      Spd xtx, ivar;
      Vec xtu, mean;

      void get_moments();
      void accumulate_moments(Ptr<Subject>);
    };
    //============================================================
    class DafePcrSubject : public PosteriorSampler{
    public:
      DafePcrSubject(Ptr<Subject> sub, Ptr<SubjectPrior> prior,
		     Ptr<DafePcrDataImputer> imp, double Tdf= -1.0);

      double logpri()const;
      void draw();
    private:
      Ptr<Subject> subject;
      Ptr<SubjectPrior> pri;
      Ptr<DafePcrDataImputer> imp;
      Ptr<MetropolisHastings> sampler;
      Ptr<MvtIndepProposal> prop;
      const double sigsq;
      Vec mean;
      Spd Ivar;
      void set_moments();
      void accumulate_moments(std::pair<Ptr<Item>, Response>);
    };


  }
}

#endif// BOOM_DAFE_PCR_HPP
