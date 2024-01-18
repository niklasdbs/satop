#include "Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>

Utils::Utils()= default;

std::vector<std::vector<std::string>> Utils::read_csv(const std::string &file_path) {
    std::ifstream file(file_path);
    std::string line;
    std::string cell;
    std::vector<std::vector<std::string>> objects;
    if (!file.is_open()) {
        throw std::ifstream::failure("fail to open file(path:" + file_path + ")");
    }
    std::getline(file, line);
    //int number_of_lines = 0;
    while (std::getline(file, line)) {
        std::istringstream line_s(line);
        std::vector<std::string> object;
        while (getline(line_s, cell, ',')) {
            object.push_back(cell);
        }

        // This checks for a trailing comma with no data after it.
        if (!line_s && cell.empty())
        {
            // If there was a trailing comma then add an empty element.
            object.push_back("");
        }

        objects.push_back(object);

        //number_of_lines++;
    }
    file.close();
    //return {objects, number_of_lines};
    return objects;
}


float Utils::average(std::vector<int> const& v){
    if(v.empty()){
        return 0.0f;
    }

    auto const count = static_cast<float>(v.size());
    return static_cast<float>(std::reduce(v.begin(), v.end())) / count;
}

float Utils::average(std::vector<float> const& v){
    if(v.empty()){
        return 0.0f;
    }

    auto const count = static_cast<float>(v.size());
    return std::reduce(v.begin(), v.end()) / count;
}

float Utils::average(std::list<int> const& v){
    if(v.empty()){
        return 0.0f;
    }

    auto const count = static_cast<float>(v.size());
    return static_cast<float>(std::reduce(v.begin(), v.end())) / count;
}

float Utils::average(std::list<float> const& v){
    if(v.empty()){
        return 0.0f;
    }

    auto const count = static_cast<float>(v.size());
    return std::reduce(v.begin(), v.end()) / count;
}