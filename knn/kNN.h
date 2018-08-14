//
// Created by a-chausov on 08.08.18.
//

#ifndef KNN_KNN_H
#define KNN_KNN_H

#include <algorithm>
#include <iostream>
#include "dataset.h"
#include "functions.h"

namespace kNN
{
    struct Model
    {
    public:
        Model() = default;
        virtual void fit(DatasetPtr const &, unsigned n_classes) = 0;
        virtual DatasetPtr test(DatasetPtr const &) = 0;
    };

    struct WeightedClassifier : Model
    {
    public:
        WeightedClassifier(unsigned n_neighbors,
                      distance::distance_func const & distance_measure,
                      kernel::kernel_func const & kernel)
                : n_neighbors(n_neighbors),
                  distance_measure(distance_measure),
                  kernel(kernel),
                  n_classes(0)
        {}

        void fit(DatasetPtr const & x, unsigned n_classes) override
        {
            this->state = x;
            this->n_classes = n_classes;
        }

        DatasetPtr test(DatasetPtr const & x) override
        {
            DatasetPtr ret(new Dataset());
            for (auto & sample : *x)
            {
                ret->emplace_back(test_one(sample), sample.coords);
            }
            return ret;
        }

        int test_one(Point const & sample)
        {
            std::vector< std::pair<double, int> > distances;
            for (size_t idx = 0; idx < this->state->size(); ++idx)
            {
                distances.emplace_back(this->distance_measure(sample, this->state->at(idx)),
                                       (this->state->at(idx).label));
            }
            std::sort(distances.begin(), distances.end());
            return apply_weights(distances);
        }

    private:
        unsigned n_neighbors;
        distance::distance_func distance_measure;
        kernel::kernel_func kernel;
        unsigned n_classes;

        DatasetPtr state;

        int apply_weights(std::vector< std::pair<double, int> > const & distances)
        {
            std::vector<double> possible_labels(n_classes);
            double max_dist = distances.at(n_neighbors - 1).first;
            for (size_t idx = 0; idx < this->n_neighbors; ++idx)
            {
                double normalized_dist = distances[idx].first / max_dist;
                possible_labels[distances[idx].second] += normalized_dist;
            }

            double max_possible = -1;
            int max_label;
            for (size_t idx = 0; idx < this->n_classes; ++idx)
            {
                if (possible_labels[idx] > max_possible)
                {
                    max_possible = possible_labels[idx];
                    max_label = idx;
                }
            }
            return max_label;
        }
    };
};

#endif //KNN_KNN_H
