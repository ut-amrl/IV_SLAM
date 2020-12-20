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

#include "dataset_creator.h"
#include "io_access.h"

#include <sys/stat.h>
#include <dirent.h>

#include <glog/logging.h>
// #include <bitset>

namespace feature_evaluation {

using std::string;

using namespace ORB_SLAM2;


DatasetCreator::DatasetCreator( const string& dataset_path)
                                : dataset_path_(dataset_path) 
{
  if( !CreateDirectory(dataset_path_) ) 
  {
    LOG(FATAL) << "Could not create the directory for saving the descriptors";
  }
}

void DatasetCreator::SaveToFile() 
{
  // Write the keypoints information to file.
  if( !WriteJsonToFile( dataset_path_,
                        "/"+keypoints_file_name_,
                        keypoints_json_) ) 
  {
    LOG(FATAL) << "Could not create directory "
              << dataset_path_ + "/" + keypoints_file_name_;
  }
  
  // Write the image names to file.
  if( !WriteJsonToFile( dataset_path_,
                        "/"+img_names_file_name_,
                        img_names_json_) ) 
  {
    LOG(FATAL) << "Could not create directory "
              << dataset_path_ + "/" + img_names_file_name_;
  }

  return;
}

void DatasetCreator::AppendKeypoints( std::vector<cv::KeyPoint> keypoints,
                                      std::vector<float> epipolar_err ) 
{
  for(size_t i = 0; i < keypoints.size(); ++i) 
  {
    keypoints_json_["x_coord"].append( keypoints[i].pt.x );
    keypoints_json_["y_coord"].append( keypoints[i].pt.y );
    keypoints_json_["response"].append( keypoints[i].response );
    keypoints_json_["size"].append( keypoints[i].size );
    keypoints_json_["epipolar_err"].append( epipolar_err[i] );
  }
 
  Json::Value keypt_id_obj;
  for( size_t i = 0; i < keypoints.size(); ++i ) 
  {
    keypt_id_obj["keypt_id"].append( (int)(i + keypt_counter_) );
  }

  img_names_json_["corresponding_keypt_id"].append( keypt_id_obj );
  keypt_counter_ += keypoints.size();

  return;
}

// NOTE: Currently, it is assumed that one uses either "AppendDescriptors" or
// "SaveBadRegionHeatmap" for creating a dataset and not both together. List
// of image names is being appended in both.
void DatasetCreator::AppendDescriptors( cv::Mat descriptors,
                                        cv::Mat descriptors2,
                                        string& img_name ) 
{
  string descriptor_file_path = dataset_path_ + "/" + descriptors_file_name_;
  string descriptor2_file_path = dataset_path_ + "/" + descriptors_2_file_name_;
  if( !CreateDirectory(dataset_path_ ) ) 
  {
    LOG(FATAL) << "Could not create the direcotory for saving the descriptors";
  }
  
  // Saves the descriptors of the first image to file
  std::ofstream myfile;
  myfile.open( descriptor_file_path.c_str(),
               std::ofstream::out | std::ofstream::app );
  
  // Save descriptors as a string of binary
  //   for(int i = 0; i < descriptors.rows; i++) {
  //     std::string binary_string = "";
  //     for(int j = 0; j < descriptors.cols; j++) {
  //       binary_string += std::bitset<8>(descriptors.at<uchar>(i,j)).to_string();
  //     }
  //     myfile << binary_string << std::endl;
  //   }
  
  myfile << cv::format( descriptors, cv::Formatter::FMT_CSV ) << std::endl;
  myfile.close();
  
  // Saves the matching descriptors on the second image to file
  std::ofstream myfile2;
  myfile2.open( descriptor2_file_path.c_str(),
                std::ofstream::out | std::ofstream::app );
  myfile2 << cv::format( descriptors2, cv::Formatter::FMT_CSV ) << std::endl;
  myfile2.close();
  
  img_names_json_["img_name"].append( img_name );
  
  Json::Value desc_id_obj;
  for( size_t i = 0; i < descriptors.rows; ++i) 
  {
    desc_id_obj["descriptor_id"].append( (int)(i + descriptor_counter_) );
  }

  img_names_json_["corresponding_descriptor_id"].append( desc_id_obj );
  descriptor_counter_ += descriptors.rows;

  return;
}

void DatasetCreator::SaveBadRegionHeatmap( const string& img_name,
                                           const cv::Mat& bad_region_heatmap ) 
{
  string img_dir = dataset_path_ + "/bad_region_heatmap/";
  if( !CreateDirectory(img_dir) ) {
    LOG(FATAL) << "Could not create the directory for saving the heatmaps";
  }

  string img_path = img_dir + img_name;
  cv::imwrite( img_path, bad_region_heatmap );
  
  img_names_json_["img_name"].append( img_name );

  return;
}

void DatasetCreator::SaveBadRegionHeatmapMask( const string& img_name,
                                               const cv::Mat& bad_region_heatmap_mask ) 
{
  string img_dir = dataset_path_ + "/bad_region_heatmap_mask/";
  if( !CreateDirectory(img_dir) ) {
    LOG(FATAL) << "Could not create the directory for saving the heatmaps";
  }

  string img_path = img_dir + img_name;
  cv::imwrite( img_path, bad_region_heatmap_mask );

  return;
}

} // namespace feature_evaluation
