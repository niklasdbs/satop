#pragma once
#include <vector>
#include "Node.h"

class Edge{
public:
    Edge(int id, float length, Node* source, Node* target);
    const float id;
    const int length;
    const Node* const source;
    const Node* const target;
    std::vector<int> resourceIDs;
};
