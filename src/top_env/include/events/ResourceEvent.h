#pragma once
#include "Event.h"
#include <string>
#include <vector>

enum ResourceEventType
{
    arrival = 0,
    depature = 1,
    violation = 2,
    start = 3 //to indicate a virtual event at the start
};

class ResourceEvent : public Event
{
public:
    ResourceEvent(std::vector<std::string> &csv);
    inline ResourceEvent(const int resourceID,
                  ResourceEventType resourceEventType,
                  const int maxSeconds,
                  const int year,
                  const int month,
                  const int day,
                  const int hour,
                  const int dayOfWeek,
                  const int time_of_day,
                  const int time)
        : resourceID(resourceID),
          resourceEventType(resourceEventType),
          maxSeconds(maxSeconds),
          year(year),
          month(month),
          day(day), hour(hour), dayOfWeek(dayOfWeek), time_of_day(time_of_day), Event(time, resource_event){};
    const int resourceID;
    ResourceEventType resourceEventType;
    const int maxSeconds;
    const int year;
    const int month;
    const int day;
    const int hour;
    const int dayOfWeek;
    const int time_of_day;
};