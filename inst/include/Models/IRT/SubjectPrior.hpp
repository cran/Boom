/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_SUBJECT_PRIOR_HPP
#define BOOM_SUBJECT_PRIOR_HPP

#include <Models/IRT/Subject.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{
  class MvnModel;
  class MvRegModel;
  namespace IRT{

    class SubjectPrior
      : virtual public Model
    {
    public:
      virtual SubjectPrior * clone()const=0;
      virtual double pdf(Ptr<Data>, bool logsc)const=0;
      virtual double pdf(Ptr<Subject>, bool logsc)const=0;
      virtual Vec mean(Ptr<Subject>)const=0;
      virtual Spd siginv()const=0;
      virtual void add_data(Ptr<Data>)=0;
      virtual void add_data(Ptr<Subject>)=0;
    };
    //------------------------------------------------------------
    class MvnSubjectPrior
      : public SubjectPrior,
	public CompositeParamPolicy,
	public IID_DataPolicy<Subject>,
	public PriorPolicy
    {
    public:
      explicit MvnSubjectPrior(Ptr<MvnModel> Mvn);
      MvnSubjectPrior(const MvnSubjectPrior &rhs);
      virtual MvnSubjectPrior * clone()const;

      virtual double pdf(Ptr<Data>, bool logsc)const;
      virtual double pdf(Ptr<Subject>, bool logsc)const;
      virtual void initialize_params();
      virtual void clear_data();
      virtual void add_data(Ptr<Data>);
      virtual void add_data(Ptr<Subject>);
      virtual Vec mean(Ptr<Subject>)const;
      virtual Spd siginv()const;
   private:
      Ptr<MvnModel> mvn;
    };
    //------------------------------------------------------------
    class MvRegSubjectPrior
      : public SubjectPrior{
    public:
      explicit MvRegSubjectPrior(Ptr<MvRegModel> mvr);
      MvRegSubjectPrior(const MvRegSubjectPrior &rhs);
      virtual MvRegSubjectPrior * clone()const;

      virtual double pdf(Ptr<Data>, bool logsc)const;
      virtual double pdf(Ptr<Subject>, bool logsc)const;
      virtual void initialize_params();
      virtual void add_data(Ptr<Data>);
      virtual void add_data(Ptr<Subject>);
      virtual Vec mean(Ptr<Subject>)const;
      virtual Spd siginv()const;
    private:
      Ptr<MvRegModel> mvreg_;
    };
  }
}
#endif// BOOM_SUBJECT_PRIOR_HPP
