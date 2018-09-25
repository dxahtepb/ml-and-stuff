//
// Created by a-chausov on 08.08.18.
//

#ifndef KNN_DATASET_H
#define KNN_DATASET_H

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <ostream>


namespace kNN
{
    struct Point;
    using Dataset = std::vector<Point>;
    using DatasetPtr = std::shared_ptr<Dataset>;
    using ConfusionMatrix = std::unordered_map<std::string, long>;
    using ConfusionMatrixPtr = std::shared_ptr<ConfusionMatrix>;

    struct Point
    {
    public:
        unsigned label;
        std::vector<double> coords;

        template <class ... Types>
        explicit
        Point(unsigned label, Types ... coords)
                : label(label)
        {
            std::array<double, sizeof ...(Types)> const coords_array {coords...};
            for (auto const coord : coords_array)
            {
                this->coords.push_back(coord);
            }
        }

        Point(unsigned label, std::vector<double> coords)
                : label(label), coords(std::move(coords))
        {}

        friend std::ostream &
        operator<< (std::ostream &output, Point const &point)
        {
            std::string coord_repr;
            for (double coord : point.coords)
            {
                coord_repr.append(" ");
                coord_repr.append(std::to_string(coord));
            }
            return output << "(coords:" << coord_repr << ", label: " << point.label << ")";
        }
    };
}

#endif //KNN_DATASET_H
