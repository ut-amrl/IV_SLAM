// Copyright 2019 srabiee@cs.umass.edu
// College of Information and Computer Sciences,
// University of Massachusetts Amherst
//
//
// This software is free: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License Version 3,
// as published by the Free Software Foundation.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// Version 3 in the file COPYING that came with this distribution.
// If not, see <http://www.gnu.org/licenses/>.
// ========================================================================
#ifndef iSLAM_DATASET_CREATOR
#define iSLAM_DATASET_CREATOR

#include <vector>
#include <algorithm>
#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <jsoncpp/json/json.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"

#include "io_access.h"


namespace feature_evaluation {

class DatasetCreator{
public:
  DatasetCreator(const std::string& dataset_path);
  ~DatasetCreator() = default;

  void SaveToFile();
  void AppendKeypoints(std::vector<cv::KeyPoint> keypoints,
                       std::vector<float> epipolar_err);
  void AppendDescriptors(cv::Mat descriptors,
                         cv::Mat descriptors2,
                        std::string& img_name);
  void SaveBadRegionHeatmap(const std::string& img_name,
                            const cv::Mat& bad_region_heatmap);
  void SaveBadRegionHeatmapMask(const std::string& img_name,
                                const cv::Mat& bad_region_heatmap_mask);
  
private:
  const std::string keypoints_file_name_ = "keypoints.json";
  const std::string descriptors_file_name_ = "descriptors.csv";
  const std::string descriptors_2_file_name_ = "descriptors_2.csv";
  const std::string img_names_file_name_ = "img_names.json";
  
  std::string dataset_path_;
  Json::Value keypoints_json_;
  Json::Value img_names_json_;
  std::vector<Eigen::VectorXd> descriptors_;
  long unsigned int descriptor_counter_ = 0;
  long unsigned int keypt_counter_ = 0;
  
 

};
} // namespace feature_evaluation

#endif // iSLAM_DATASET_CREATOR

