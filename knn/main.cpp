#include <iostream>
#include <fstream>
#include <sstream>
#include "kNN.h"


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
read_csv(std::string const & file_name,
         std::vector< std::vector<std::string> > & table,
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
split_dataset(kNN::DatasetPtr & all_set,
              kNN::DatasetPtr & train_set,
              kNN::DatasetPtr & test_set,
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


const double RATIO = 0.8;
const unsigned SEED = 1488;
const std::string DATASET_FILE = "dataset.txt";


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
        double x = std::stod(row[0]), y = std::stod(row[1]);
        int label = std::stoi(row[2]);
        all_samples->emplace_back(kNN::Point(label, x, y));
    }

    kNN::DatasetPtr train_set(new kNN::Dataset());
    kNN::DatasetPtr test_set(new kNN::Dataset());
    kNN::DatasetPtr predicted_set;

    err = split_dataset(all_samples, train_set, test_set, RATIO, SEED);

    for (unsigned k = 1; k < 20; k += 1)
    {
        kNN::WeightedClassifier classifier(k, kNN::distance::euclidean, kNN::kernel::uniform);
        classifier.fit(train_set, 2);
        predicted_set = classifier.test(test_set);

        auto matrix = kNN::evaluate_confusion_matrix(*test_set, *predicted_set);
        std::cout << k << " " << kNN::evaluate_F1_score(matrix) << std::endl;
    }

    return 0;
}