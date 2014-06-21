#ifndef BOOM_RESAMPLER_HPP
#define BOOM_RESAMPLER_HPP
#include <BOOM.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Types.hpp>
#include <vector>
#include <map>
#include <algorithm>

namespace BOOM{


  class Resampler{
    // efficiently samples values from 0..Nvals()-1 with replacement
    // from the discrete distribution specified by probs.

    // a vector<Things> is resampled by calling std::vector<uint> indx = operator(N);

  public:
    Resampler(uint nvals=1); // equally weighted [0..nvals-1]
    Resampler(const std::vector<double> &probs, bool normalize=true); // nvals determined by Probs

    template <class T>
    std::vector<T> operator()(const std::vector<T> &) const;
    std::vector<uint> operator()(uint N)const;

    uint Nvals()const;
    void set_probs(const std::vector<double> &probs, bool normalize=true);

  private:
    typedef std::map<double, uint> CDF;
    CDF cdf;
    void setup_cdf(const std::vector<double> &probs, bool normalize=true);
    void flush_cdf();
  };

  //------------------------------------------------------------

  template <class T>
  std::vector<T> Resampler::operator()(const std::vector<T> &things)const{
    uint N = things.size();
    std::vector<uint> indx = (*this)(N);
    std::vector<T> ans;
    ans.reserve(N);
    for(uint i=0; i<N; ++i) ans[i] = things[indx[i]];
    return ans;
  }

  //______________________________________________________________________

  template <class T>
  std::vector<T> resample(const std::vector<T> &, uint n, const Vec & probs);

  template<class T>
  std::vector<T> resample(const std::vector<T> & things, uint Nthings, const Vec & probs){
    Vec cdf = cumsum(probs);
    double total = cdf.back();
    if(total<1.0 || total > 1.0) {
      cdf/=total;
      total=1.0;
    }

    Vec u(Nthings);
    u.randomize();
    std::sort(u.begin(), u.end());

    std::vector<T> ans;
    ans.reserve(Nthings);
    uint cursor=0;
    for(uint i=0; i<Nthings; ++i){
      while(u[i]>cdf[cursor]) ++cursor;
      ans.push_back(things[cursor]);
    }
    return(ans);
  }
}
#endif// BOOM_RESAMPLER_HPP
