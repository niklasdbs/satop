#pragma once
#include <map>
#include <list>
#include <variant>
#include <string>

#define type_in_var float, int, bool, std::string, std::list<std::string>, std::list<int>, std::list<float>, std::list<bool>
using Config = std::map<std::string, std::variant<type_in_var, std::map<std::string, std::variant<type_in_var, std::map<std::string, std::variant<type_in_var, std::map<std::string, std::variant<type_in_var>>>>>>>>;
