//
// Created by a-chausov on 22.08.18.
//

#ifndef KNN_STATISTICS_H
#define KNN_STATISTICS_H

#include "dataset.h"

namespace kNN
{
    std::shared_ptr<std::vector<std::vector<int>>>
    fill_prediction_matrix(DatasetPtr const &known_classes,
                           DatasetPtr const &predicted_classes,
                           unsigned n_classes)
    {
        std::shared_ptr<std::vector<std::vector<int>>> confusion(
        new std::vector<std::vector<int>>);
        confusion->resize(n_classes);
        for (auto &x : *confusion)
        {
            x.resize(n_classes);
        }

        for (size_t idx = 0; idx < predicted_classes->size(); ++idx)
        {
            auto pred_class = static_cast<size_t> (predicted_classes->at(idx).label);
            auto true_class = static_cast<size_t> (known_classes->at(idx).label);
            confusion->at(pred_class).at(true_class)++;
        }
        return confusion;
    }

    ConfusionMatrixPtr
    evaluate_confusion_matrix(DatasetPtr const & known_classes,
                              DatasetPtr const & predicted_classes,
                              unsigned n_classes, unsigned cls)
    {
        std::shared_ptr<ConfusionMatrix> confusion_matrix(new ConfusionMatrix());
        for (auto const & cell : {"TP", "TN", "FP", "FN"})
        {
            confusion_matrix->insert({cell, 0});
        }

        auto confusion = fill_prediction_matrix(known_classes, predicted_classes, n_classes);

        confusion_matrix->at("TP") = confusion->at(cls).at(cls);
        for (size_t idx = 0; idx < n_classes; ++idx)
        {
            if (idx == cls)
            {
                continue;
            }
            confusion_matrix->at("FP") += confusion->at(cls).at(idx);
            confusion_matrix->at("FN") += confusion->at(idx).at(cls);
        }
        for (size_t idx_i = 0; idx_i < n_classes; ++idx_i)
        {
            for (size_t idx_j = 0; idx_j < n_classes; ++idx_j)
            {
                if (idx_i == cls || idx_j == cls)
                {
                    continue;
                }
                confusion_matrix->at("TN") += confusion->at(idx_i).at(idx_j);
            }
        }

        return confusion_matrix;
    }


    ConfusionMatrixPtr
    evaluate_confusion_matrix(DatasetPtr const & known_classes,
                              DatasetPtr const & predicted_classes)
    {
        return evaluate_confusion_matrix(known_classes, predicted_classes, 2, 1);
    }


    double
    evaluate_acc(DatasetPtr const &known_classes,
                 DatasetPtr const &predicted_classes)
    {
        unsigned long guessed = 0, missed = 0;
        for (size_t idx = 0; idx < known_classes->size(); ++idx)
        {
            if (known_classes->at(idx).label == predicted_classes->at(idx).label)
            {
                ++guessed;
            } else
            {
                ++missed;
            }
        }
        return guessed * 1.0f / (guessed + missed);
    }

    double calculate_recall(ConfusionMatrixPtr const & confusion)
    {
        return 1.0f * confusion->at("TP") / (confusion->at("TP") + confusion->at("FN"));
    }

    double
    calculate_recall(DatasetPtr const & known_classes, DatasetPtr const & predicted_classes)
    {
        auto confusion = evaluate_confusion_matrix(known_classes, predicted_classes);

        return calculate_recall(confusion);
    }

    double
    evaluate_F1_score(ConfusionMatrixPtr const & confusion)
    {
        return 2.0f * confusion->at("TP") /
               (2.0f * confusion->at("TP") + confusion->at("FP") + confusion->at("FN"));
    }

    double
    evaluate_F1_score(DatasetPtr const & known_classes, DatasetPtr const & predicted_classes)
    {
        ConfusionMatrixPtr confusion = evaluate_confusion_matrix(known_classes,
                                                                 predicted_classes);
        return evaluate_F1_score(confusion);
    }

    double calculate_precision(ConfusionMatrixPtr const & confusion)
    {
        return 1.0f * confusion->at("TP") / (confusion->at("TP") + confusion->at("FP"));
    }

    double
    calculate_precision(DatasetPtr const & known_classes, DatasetPtr const & predicted_classes)
    {
        auto confusion = evaluate_confusion_matrix(known_classes, predicted_classes);

        return calculate_precision(confusion);
    }
}



#endif //KNN_STATISTICS_H
