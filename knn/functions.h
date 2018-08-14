//
// Created by a-chausov on 08.08.18.
//

#ifndef KNN_FUNCTIONS_H
#define KNN_FUNCTIONS_H

#include "dataset.h"
#include <cmath>

namespace kNN
{
    namespace distance
    {
        using distance_func = std::function<double(kNN::Point, kNN::Point)>;

        template<int P>
        distance_func minkowski = [](Point const & a, Point const & b) {
            long double mink_sum = 0;
            for (size_t idx = 0; idx < a.coords.size(); ++idx)
            {
                mink_sum += pow(fabs(a.coords[idx] - b.coords[idx]), P);
            }
            return static_cast<double> (powl(mink_sum, 1.0/P));
        };

        distance_func chebyshev = [](Point const & a, Point const & b) {
            double max_dist = -1;
            for (size_t idx = 0; idx < a.coords.size(); ++idx)
            {
                max_dist = std::max(max_dist, fabs(a.coords[idx] - b.coords[idx]));
            }
            return max_dist;
        };

        distance_func euclidean = minkowski<2>;
        distance_func manhattan = minkowski<1>;
    }

    namespace kernel
    {
        using kernel_func = std::function<double(double)>;

        kernel_func uniform = [](double u) -> double {
            return 0.5;
        };

        kernel_func gaussian = [](double u) -> double {
            return std::pow(M_E, -u*u / 2) / sqrt(2 * M_PI);
        };

        kernel_func sigmoid = [](double u) -> double {
            return 2 / (M_PI * (std::pow(M_E, u) + std::pow(M_E, -u)));
        };

        kernel_func triangular = [](double u) -> double {
            return 1 - fabs(u);
        };

        kernel_func quartic = [](double u) -> double {
            return 15 * ((1 - u*u)*(1 - u*u)) / 16;
        };

        kernel_func epanechnikov = [](double u) -> double {
            return 3 * (1 - u*u) / 4;
        };

        kernel_func logistic = [](double u) -> double {
            return 1 / (std::pow(M_E, u) + 2 + std::pow(M_E, -u));
        };

        kernel_func cosine = [](double u) -> double {
            return M_PI_4 * std::cos(M_PI_2 * u);
        };
    }
}

#endif //KNN_FUNCTIONS_H
