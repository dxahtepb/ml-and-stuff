//
// Created by a-chausov on 08.08.18.
//

#ifndef KNN_KNN_H
#define KNN_KNN_H

#include <algorithm>
#include <iostream>
#include <utility>

#include "dataset.h"
#include "statistics.h"
#include "functions.h"

namespace kNN
{
    struct ClassifierModel
    {
    public:
        ClassifierModel() = default;
        virtual void fit(DatasetPtr const &, unsigned n_classes) = 0;
        virtual DatasetPtr test(DatasetPtr const &) = 0;
    };

    struct WeightedClassifier : ClassifierModel
    {
    public:
        WeightedClassifier(unsigned n_neighbors,
                           distance::distance_func distance_measure,
                           kernel::kernel_func kernel)
                : n_neighbors(n_neighbors),
                  distance_measure(std::move(distance_measure)),
                  kernel(std::move(kernel)),
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

        long test_one(Point const & sample_point)
        {
            std::vector< std::pair<double, unsigned> > distances;
            for (auto &known_point : *this->state)
            {
                distances.emplace_back(this->distance_measure(sample_point, known_point),
                                       known_point.label);
            }
            std::sort(distances.begin(), distances.end());
            return apply_weights(distances);
        }

    private:
        unsigned long n_neighbors;
        distance::distance_func distance_measure;
        kernel::kernel_func kernel;
        unsigned long n_classes;

        DatasetPtr state;

        long apply_weights(std::vector< std::pair<double, unsigned> > const & distances)
        {
            std::vector<double> possible_labels(n_classes, 0);

            double max_dist = distances.size() <= n_neighbors ?
                    distances.end()->first : distances.at(n_neighbors - 1).first;

            for (size_t idx = 0; idx < n_neighbors && idx < distances.size(); ++idx)
            {
                double normalized_dist = distances[idx].first / max_dist;
                possible_labels[distances[idx].second] += kernel(normalized_dist);
            }

            double max_possible = -1;
            long max_label = 0;
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
