//
// Created by a-chausov on 23.08.18.
//

#ifndef KNN_PLOTTING_H
#define KNN_PLOTTING_H

#include "dataset.h"
#include "gnuplot-iostream.h"

void plot3d(kNN::DatasetPtr &test_set, kNN::DatasetPtr &train_set,
            kNN::DatasetPtr &predicted_set)
{
    std::vector<boost::tuples::tuple<double, double, double, double> > xyz_pts_0_cls;
    std::vector<boost::tuples::tuple<double, double, double, double> > xyz_pts_1_cls;
    for (auto &points_set : {test_set, train_set})
    {
        for (auto const &point : *points_set.get())
        {
            double point_size = &(*points_set) == &(*train_set) ? 0.03f : 0.05f;
            auto *points_vector = point.label == 0 ? &xyz_pts_0_cls : &xyz_pts_1_cls;
            points_vector->emplace_back(point.coords[0], point.coords[1], point.coords[2], point_size);
        }
    }

    std::vector<boost::tuples::tuple<double, double, double, double> > xyz_pts_predict_false;
    for (size_t idx = 0; idx < predicted_set->size(); ++idx)
    {
        if (predicted_set->at(idx).label != test_set->at(idx).label)
        {
            xyz_pts_predict_false.emplace_back(predicted_set->at(idx).coords[0],
                                               predicted_set->at(idx).coords[1],
                                               predicted_set->at(idx).coords[2],
                                               0.025f);
        }
    }

    Gnuplot gp;
    gp << "set terminal wxt size 800,600\n"
       << "set xrange [-1.5:1.5]\nset yrange [-1.5:1.5]\n"
       << "splot" << gp.file1d(xyz_pts_0_cls) << "with points lc rgb 'green' title 'class 0',"
       << gp.file1d(xyz_pts_1_cls) << "with points lc rgb 'blue' title 'class 1',"
       << gp.file1d(xyz_pts_predict_false) << "with points lc rgb 'red' title 'mispredicted'"
       << std::endl;
}


void plot2d(kNN::DatasetPtr &test_set, kNN::DatasetPtr &train_set,
            kNN::DatasetPtr &predicted_set)
{
    std::vector<boost::tuples::tuple<double, double, double> > xy_pts_0_cls;
    std::vector<boost::tuples::tuple<double, double, double> > xy_pts_1_cls;
    for (auto &points_set : {test_set, train_set})
    {
        for (auto const &point : *points_set.get())
        {
            double point_size = &(*points_set) == &(*train_set) ? 0.03f : 0.05f;
            auto *points_vector = point.label == 0 ? &xy_pts_0_cls : &xy_pts_1_cls;
            points_vector->emplace_back(point.coords[0], point.coords[1], point_size);
        }
    }

    std::vector<boost::tuples::tuple<double, double, double> > xy_pts_predict_false;
    for (size_t idx = 0; idx < predicted_set->size(); ++idx)
    {
        if (predicted_set->at(idx).label != test_set->at(idx).label)
        {
            xy_pts_predict_false.emplace_back(predicted_set->at(idx).coords[0],
                                              predicted_set->at(idx).coords[1],
                                              0.025f);
        }
    }

    Gnuplot gp;
    gp << "set xrange [-1.5:1.5]\nset yrange [-1.5:1.5]\n"
       << "set style fill transparent solid 0.5 noborder\n"
       << "set style circles radius 0.05\n"
       << "plot" << gp.file1d(xy_pts_0_cls) << "with circles lc rgb 'green' title 'class 0',"
       << gp.file1d(xy_pts_1_cls) << "with circles lc rgb 'blue' title 'class 1',"
       << gp.file1d(xy_pts_predict_false) << "with circles lc rgb 'red' title 'mispredicted'"
       << std::endl;
}

#endif //KNN_PLOTTING_H
