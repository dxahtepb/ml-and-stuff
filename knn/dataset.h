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
        int label = 0;
        std::vector<double> coords;

        template <class ... Types>
        explicit
        Point(int label, Types ... coords)
        {
            std::array<double, sizeof ...(Types)> const coords_array {coords...};
            for (auto const coord : coords_array)
            {
                this->coords.push_back(coord);
            }
            this->label = label;
        }

        Point(int label, std::vector<double> const & coords) : label(label)
        {
            this->coords = coords;
        }

        bool operator== (Point const & other) const
        {
            if (this->coords.size() != other.coords.size())
            {
                return false;
            }
            for (size_t idx = 0; idx < this->coords.size(); ++idx)
            {
                if (this->coords[idx] != other.coords[idx])
                {
                    return false;
                }
            }
            return true;
        }

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

    ConfusionMatrixPtr
    evaluate_confusion_matrix(Dataset const & known_classes,
                              Dataset const & predicted_classes)
    {
        std::shared_ptr<ConfusionMatrix> confusion_matrix(new ConfusionMatrix());
        for (auto const & cell : {"TP", "TN", "FP", "FN"})
        {
            confusion_matrix->insert({cell, 0});
        }

        for (auto it_known = known_classes.begin(), it_predict = predicted_classes.begin();
             it_known != known_classes.end() && it_predict != known_classes.end();
             it_known++, it_predict++)
        {
            if (it_known->label == it_predict->label)
            {
                confusion_matrix->at(it_known->label == 1 ? "TP" : "TN")++;
            } else
            {
                confusion_matrix->at(it_known->label == 1 ? "FN" : "FP")++;
            }
        }

        return confusion_matrix;
    }


    double
    evaluate_F1_score(ConfusionMatrixPtr const & confusion_matrix)
    {
        return 2.0f * confusion_matrix->at("TP") /
               (2.0f * confusion_matrix->at("TP")
                + confusion_matrix->at("FP") + confusion_matrix->at("FN"));
    }
}
#endif //KNN_DATASET_H
