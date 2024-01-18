#pragma once

enum EventType{agent_event = 0, resource_event = 1};

class Event{
public:
    explicit Event(int eventTime, EventType eventType);
    const int eventTime;
    const EventType eventType;
};