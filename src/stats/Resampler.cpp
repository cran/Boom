#include <stats/Resampler.hpp>
#include <LinAlg/Vector.hpp>
#include <distributions.hpp>
#include <stdexcept>
#include <numeric>

namespace BOOM{


  Resampler::Resampler(uint N){
    for(uint i=0; i<N; ++i){
      double p = i+1;
      p/=N;
      cdf[p] = i;
    }
  }

  Resampler::Resampler(const std::vector<double> &probs, bool normalize){
    setup_cdf(probs, normalize);
  }

  std::vector<uint> Resampler::operator()(uint N)const{
    std::vector<uint> ans(N);
    for(uint i=0; i<N; ++i){
      double u = runif();
      uint indx = cdf.lower_bound(u)->second;
      ans[i] = indx;
    }
    return ans;
  }

  void Resampler::flush_cdf(){
    CDF new_cdf;
    cdf.swap(new_cdf);
  }


  void Resampler::set_probs(const std::vector<double> &probs, bool normalize){
    flush_cdf();
    setup_cdf(probs, normalize);

  }

  void Resampler::setup_cdf(const std::vector<double> &probs, bool normalize){
    uint N = probs.size();
    double nc = 1.0;
    if(normalize)
      nc = std::accumulate(probs.begin(), probs.end(), 0.0);
    double p(0);
    for(uint i=0; i<N; ++i){
      double p0 = probs[i]/nc;
      if(p0<0) throw_exception<std::runtime_error>("negative prob");
      p+= p0;
      cdf[p] = i;
    }
  }

  uint Resampler::Nvals()const{ return cdf.size();}

}
