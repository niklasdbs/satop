#include "graph/Edge.h"
#include "graph/Node.h"

Edge::Edge(int id, float length, Node* source, Node* target) : id(id), length(length), source(source), target(target)
{
}

