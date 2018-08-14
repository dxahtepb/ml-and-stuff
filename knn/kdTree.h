//
// Created by a-chausov on 10.08.18.
//

#ifndef KNN_KDTREE_H
#define KNN_KDTREE_H

#include <memory>

namespace kNN
{
    struct KdNode
    {
        std::unique_ptr<KdNode> left;
        std::unique_ptr<KdNode> right;
    };

    struct KdTree
    {

    };
}

#endif //KNN_KDTREE_H
