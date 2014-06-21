#include <Models/HMM/Clickstream/Event.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  namespace Clickstream{

    Event::Event(int value, Ptr<CatKey> key)
        : MarkovData(value, key)
    {}

    Event::Event(int value, Ptr<Event> prev)
        : MarkovData(value, Ptr<MarkovData>(prev))
    {}

    Event * Event::clone()const{return new Event(*this);}

  }  // namespace Clickstream
}  // namespace BOOM
