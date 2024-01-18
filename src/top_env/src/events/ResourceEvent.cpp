#include "events/ResourceEvent.h"

ResourceEvent::ResourceEvent(std::vector<std::string> &csv) : Event(stoi(csv[9]), resource_event),
                                                              resourceID(stoi(csv[1])),
                                                              maxSeconds(stoi(csv[3])),
                                                              year(stoi(csv[5])),
                                                              month(stoi(csv[6])),
                                                              day(stoi(csv[7])),
                                                              hour(stoi(csv[8])),
                                                              time_of_day(stoi(csv[9])),
                                                              dayOfWeek(stoi(csv[10])),
                                                              resourceEventType(static_cast<ResourceEventType>(stoi(csv[4]))) // 0 = arrival, 1 = depature, 2 = violation
{
}