#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <list>

class Utils {
public:
    Utils();

    static std::vector<std::vector<std::string>> read_csv(const std::string &file_path);
    static float average(const std::vector<int> &v);
    static float average(const std::vector<float> &v);
    static float average(const std::list<int> &v);
    static float average(const std::list<float> &v);

};