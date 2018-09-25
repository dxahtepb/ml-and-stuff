#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "kNN.h"


const auto SEED = static_cast<unsigned> (
        std::chrono::system_clock::now().time_since_epoch().count());
const std::string DATASET_FILE = "dataset.txt";


struct ReturnCode
{
    int code;
    std::string message;

    ReturnCode()
            : code(), message()
    {}

    ReturnCode(int code, std::string message)
            : code(code), message(std::move(message))
    {}
};

ReturnCode
read_csv(std::string const &file_name,
         std::vector<std::vector<std::string> > &table,
         char delim)
{
    std::ifstream dataset_file(file_name);

    if (!dataset_file.good())
    {
        return {1, "No file \"" + file_name + "\" found"};
    }

    std::string line;
    while (std::getline(dataset_file, line))
    {
        std::vector<std::string> tokens;
        std::string token;
        std::stringstream ss(line);
        while (std::getline(ss, token, delim))
        {
            tokens.emplace_back(std::move(token));
        }
        table.emplace_back(std::move(tokens));
    }
    return {0, ""};
}

ReturnCode
split_dataset(kNN::DatasetPtr &all_set,
              kNN::DatasetPtr &train_set,
              kNN::DatasetPtr &test_set,
              double ratio,
              unsigned seed)
{
    std::shuffle(all_set->begin(), all_set->end(), std::default_random_engine(seed));

    auto train_set_size = static_cast<unsigned> (ratio * all_set->size());
    for (size_t idx = 0; idx < train_set_size; ++idx)
    {
        train_set->push_back(all_set->at(idx));
    }
    for (size_t idx = train_set_size; idx < all_set->size(); ++idx)
    {
        test_set->emplace_back(all_set->at(idx));
    }

    if (train_set->size() + test_set->size() == all_set->size())
    {
        return {0, ""};
    }
    else
    {
        return {1, "Wrong splitting"};
    }
}

ReturnCode
split_dataset_k_fold(kNN::DatasetPtr &all_set,
                     kNN::DatasetPtr &train_set,
                     kNN::DatasetPtr &test_set,
                     size_t from,
                     size_t to)
{
    test_set->clear();
    train_set->clear();
    for (size_t idx = 0; idx < all_set->size(); ++idx)
    {
        if (from <= idx && idx < to)
        {
            test_set->emplace_back(all_set->at(idx));
        } else
        {
            train_set->emplace_back(all_set->at(idx));
        }
    }

    if (train_set->size() + test_set->size() == all_set->size())
    {
        return {0, ""};
    }
    else
    {
        return {1, "Wrong splitting"};
    }
}


double k_fold_cross_validation(kNN::DatasetPtr &all_samples, kNN::ClassifierModel & classifier,
                               unsigned n_classes=2, int k_fold=10, unsigned seed=42)
{
    std::shuffle(all_samples->begin(), all_samples->end(), std::default_random_engine(seed));

    kNN::DatasetPtr train_set(new kNN::Dataset());
    kNN::DatasetPtr test_set(new kNN::Dataset());
    kNN::DatasetPtr predicted_set;

    std::function<double(kNN::DatasetPtr const &, kNN::DatasetPtr const &)> accuracy_metric;
    if (n_classes == 2)
    {
        accuracy_metric =
                [](kNN::DatasetPtr const & a, kNN::DatasetPtr const & b) -> double
                {
                    kNN::evaluate_F1_score(a, b);
                };
    }
    else
    {
        accuracy_metric = kNN::evaluate_acc;
    }

    long double acc = 0;

    for (int k = 0; k < k_fold; ++k)
    {
        size_t from = k * (all_samples->size() / k_fold);
        size_t to = (k + 1) * (all_samples->size() / k_fold);
        split_dataset_k_fold(all_samples, train_set, test_set, from, to);

        classifier.fit(train_set, n_classes);
        predicted_set = classifier.test(test_set);

        double fold_acc = accuracy_metric(test_set, predicted_set);
        acc += fold_acc;
    }

    return static_cast<double> (acc / k_fold);
}


int main()
{
    ReturnCode err;
    kNN::DatasetPtr all_samples(new kNN::Dataset());

    std::vector< std::vector<std::string> > table;
    err = read_csv(DATASET_FILE, table, ',');
    if (err.code != 0)
    {
        std::cerr << err.message << std::endl;
        return 0;
    }

    for (auto const & row : table)
    {
        std::vector<double> coords;
        for (size_t axis = 0; axis < row.size()-1; ++axis)
        {
            coords.push_back(std::stod(row[axis]));
        }
        auto label = static_cast<unsigned> (std::stoul(row[row.size()-1]));
        all_samples->emplace_back(kNN::Point(label, coords));
    }

    unsigned n_tries = 10;
    double validation_acc = 0;
    for (int _ = 0; _ < n_tries; ++_)
        for (unsigned k_neighbors = 10; k_neighbors <= 10; ++k_neighbors)
        {
            kNN::WeightedClassifier classifier(k_neighbors, kNN::distance::euclidean,
                                               kNN::kernel::epanechnikov);
            double validation = k_fold_cross_validation(all_samples, classifier, 2, 5, SEED);
            std::cout << "k: " << k_neighbors << ", metric: " << validation << std::endl;
            validation_acc += validation;
        }
    std::cout << "avg result: " << validation_acc/n_tries << std::endl;

    return 0;
}
