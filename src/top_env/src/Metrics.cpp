#include "Metrics.h"
#include "Utils.h"

std::map<std::string,float> Metrics::full_reset(){
    soft_reset();
    std::map<std::string,float> metrics =  {
        {"time_until_fine", Utils::average(time_until_fine_after_soft_reset)},
        {"violation_durations", Utils::average(violation_durations_after_soft_reset)},
        {"violation_durations_non_fined_resources", Utils::average(violation_durations_non_fined_resources_after_soft_reset)},
        {"violation_catched_quota", Utils::average(violation_catched_quota_after_soft_reset)},
        {"fined_resources", Utils::average(fined_resources_after_soft_reset)},
        {"cumulative_resources_in_violation", Utils::average(cumulative_resources_in_violation_quota_after_soft_reset)},
    };

    time_until_fine_after_soft_reset.clear();
    violation_catched_quota_after_soft_reset.clear();
    fined_resources_after_soft_reset.clear();
    cumulative_resources_in_violation_quota_after_soft_reset.clear();
    violation_durations_after_soft_reset.clear();
    violation_durations_non_fined_resources_after_soft_reset.clear();

    return metrics;
}

void Metrics::soft_reset(){
    if (cumulative_resources_in_violation>0){
        float violation_catched_quota = static_cast<float>(fined_resources) / cumulative_resources_in_violation;
        violation_catched_quota_after_soft_reset.push_back(violation_catched_quota);
        fined_resources_after_soft_reset.push_back(fined_resources);
        cumulative_resources_in_violation_quota_after_soft_reset.push_back(cumulative_resources_in_violation);

        time_until_fine_after_soft_reset.push_back(Utils::average(time_until_fine));
        violation_durations_after_soft_reset.push_back(Utils::average(violation_durations));
        violation_durations_non_fined_resources_after_soft_reset.push_back(Utils::average(violation_durations_non_fined_resources));
    }

    time_until_fine.clear();
    violation_durations.clear();
    violation_durations_non_fined_resources.clear();
    fined_resources = 0;
    cumulative_resources_in_violation = 0;

}