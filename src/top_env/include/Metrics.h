#pragma once
#include <list>
#include <string>
#include <map>

class Metrics{
public:
    std::list<int> time_until_fine;
    int fined_resources = 0;
    std::list<int> violation_durations;
    std::list<int> violation_durations_non_fined_resources;
    int cumulative_resources_in_violation = 0;
    
    //return aggregated metrics
    std::map<std::string,float> full_reset();
    /// @brief end of day
    void soft_reset();
private:
    std::list<float> time_until_fine_after_soft_reset;
    std::list<float> violation_catched_quota_after_soft_reset;
    std::list<int> fined_resources_after_soft_reset;
    std::list<int> cumulative_resources_in_violation_quota_after_soft_reset;
    std::list<float> violation_durations_after_soft_reset;
    std::list<float> violation_durations_non_fined_resources_after_soft_reset;
};