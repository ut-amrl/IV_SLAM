/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University 
of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <dirent.h>
#include <csignal>
#include<opencv2/core/core.hpp>


#include<System.h>
#include "io_access.h"

#if (CV_VERSION_MAJOR >= 4)
  #include<opencv2/imgcodecs/legacy/constants_c.h>
#endif

// Parameters 
const bool kEnableIntrospectionModel = false;
const bool kSaveVisualizationsToFile = true;
const bool kTrainIntrospectionModel = true;
const bool kCreateDataset = false;


// Set to true if you would like to use predicted heatmaps the same size as
// the input images for weighting the extracted keypoints.
const bool kPredictedImageQualityAvailable = false;

// If set to ture, the estimated camera pose uncertainty values are loaded 
// and passed to IV-SLAM
const bool kLoadRelativeCamPoseUncertaintyFromFile = false;

// Length of the image name suffix that should be extracted from its name
const int KImageNameSuffixLength = 23;


// Global variables
ORB_SLAM2::System *SLAM_ptr;

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
               vector<double> &vTimestamps);

// Loads images as well as the corresponding ground truth camera poses
void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      const string &strPathToPoseUncertainty,
                      const bool &load_pose_uncertainty,
                      vector<string> &vstrImageLeft,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps, 
                      vector<cv::Mat>* cam_pose_gt,
                      vector<Eigen::Vector2f>* rel_cam_pose_uncertainty);

bool GetImageQualFileNames(const std::string &directory,
                           const int &size,
                           vector<string> *vstrImageQualFilenames,
                           int * num_qual_imgs_found);

// Helper function to synchronize two vectors of sorted time stamps (ascending)
// Finds the index of the closest stamp in the second time stamp vector for
// each entry in the first vector
void SynchronizeTimeStamps(const vector<long int>time_stamps_1,
                           const vector<long int>time_stamps_2,
                           vector<int>* matches1to2,
                           vector<bool>* successful_matches);

void SignalHandler( int signal_num ) { 
   cout << "Interrupt signal is (" << signal_num << ").\n"; 
  
   // terminate program  
   if(SLAM_ptr) {
     SLAM_ptr->ShutdownMinimal();
  }
  
  cout << "Exiting the program!" << endl;
   
   exit(signal_num);   
} 

int main(int argc, char **argv)
{
    google::InstallFailureSignalHandler();
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 2;   // ERROR level logging.
    FLAGS_minloglevel = 1; // WARNING level
    FLAGS_colorlogtostderr = 1;  // Colored logging.
    FLAGS_logtostderr = true;    // Don't log to disk
    signal(SIGINT, SignalHandler);
  
    if(argc != 4 && !kTrainIntrospectionModel)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary "
        " path_to_settings path_to_sequence " << endl;
        return 1;
    }
    
    if( (argc != 8 && argc !=9) && kTrainIntrospectionModel)
    {
        cerr << endl << "Usage: ./stereo_kitti " 
        " path_to_vocabulary "
        " path_to_settings "
        " path_to_sequence " 
        " path_to_ground_truth "
        " path_to_pred_image_quality_folder "
        " save_visualization_path "
        " output_dataset_path"
        " path_to_rel_cam_pose_uncertainty[optional]" << endl;
        return 1;
    }
    
    // Path to cam pose uncertianty should be provided when 
    // kLoadRelativeCamPoseUncertaintyFromFile is set to true
    if(argc !=9 && kTrainIntrospectionModel && 
kLoadRelativeCamPoseUncertaintyFromFile){
        cerr << endl << "Usage: ./stereo_kitti " 
        " path_to_vocabulary "
        " path_to_settings "
        " path_to_sequence " 
        " path_to_ground_truth "
        " path_to_pred_image_quality_folder "
        " save_visualization_path "
        " output_dataset_path"
        " path_to_rel_cam_pose_uncertainty" << endl;
        return 1;
      
    }
    
    
    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, P_l, R_l, D_l;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["LEFT.P"] >> P_l;
    fsSettings["LEFT.R"] >> R_l;
    fsSettings["LEFT.D"] >> D_l;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];

    if(K_l.empty() ||  P_l.empty() ||  R_l.empty() 
           || D_l.empty() || 
            rows_l==0 ||  cols_l==0 )
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" 
             << endl;
        return -1;
    }

    cv::Mat M1l,M2l;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,
                            P_l.rowRange(0,3).colRange(0,3),
                            cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
  
    
    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrDepthImageFilenames;
    vector<string> vstrImageQualFilenames;
    vector<double> vTimestamps;
    vector<cv::Mat> cam_poses_gt;
    vector<Eigen::Vector2f> rel_cam_poses_uncertainty;
    // The map from left image names to the ID of corresponding relative camera 
    // pose uncertainty (from current image to next image)
    std::unordered_map<std::string, int> pose_unc_map;
    string save_visualization_path;
    string output_dataset_path;
   
   
    if (!kTrainIntrospectionModel) {
      LoadImages(string(argv[3]), vstrImageLeft, vTimestamps);
    } else {
      
      string pose_uncertainty_path;
      if (kLoadRelativeCamPoseUncertaintyFromFile) {
        pose_uncertainty_path = string(argv[8]);
      }
        
      LoadImagesWithGT(string(argv[3]), 
                       string(argv[4]),
                       string(argv[5]),
                       pose_uncertainty_path,
                       kLoadRelativeCamPoseUncertaintyFromFile,
                       vstrImageLeft,
                       vstrImageQualFilenames,
                       vTimestamps,
                       &cam_poses_gt,
                       &rel_cam_poses_uncertainty);
      save_visualization_path = string(argv[6]);
      output_dataset_path = string(argv[7]);
      
      if (kLoadRelativeCamPoseUncertaintyFromFile) {
        for (size_t i = 0; i < vstrImageLeft.size(); i++) {
          string img_name_short = vstrImageLeft[i].substr(
                            vstrImageLeft[i].length() - KImageNameSuffixLength, 
                            KImageNameSuffixLength);
          pose_unc_map[img_name_short] = i;
        }
      }
      
    }
    
    
    

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to 
    // process frames.
    bool viewer_on = true;
    bool single_threaded = false;
    bool use_BoW = true;
    ORB_SLAM2::System SLAM(argv[1],
                           argv[2],
                           ORB_SLAM2::System::MONOCULAR,
                           viewer_on,
                           kTrainIntrospectionModel,
                           kEnableIntrospectionModel,
                           kSaveVisualizationsToFile,
                           kCreateDataset,
                           single_threaded,
                           use_BoW,
                           save_visualization_path,
                           output_dataset_path);
    if (kLoadRelativeCamPoseUncertaintyFromFile) {
      LOG(FATAL) << "Loading relative camera pose uncertainty is not yet "
                 << "supported in Monocular mode!";
    }
  
    SLAM_ptr = &SLAM;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imLeftRect, imgLeftUnDist;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }
        
        cv::undistort(imLeft, 
                    imgLeftUnDist, 
                    K_l,
                    D_l);
        
        // Read the predicted quality image
        cv::Mat qual_img;
        if (kPredictedImageQualityAvailable) {
          // There might not be a image quality available for all input 
          // images. In that case just skip the missing ones with setting 
          // them to empty images. (The SLAM object will ignore empty
          // images as if no score was available).
          if(vstrImageQualFilenames[ni].empty()) {
            qual_img = cv::Mat(0, 0, CV_8U);
          } else {
          
            // Read the predicted image quality
            qual_img = cv::imread(
                            vstrImageQualFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
            if(qual_img.empty()) {
                cerr << endl << "Failed to load image at: " << 
                                vstrImageQualFilenames[ni] << endl;
                return 1;
            }
          }
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = 
std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = 
std::chrono::monotonic_clock::now();
#endif
        
        string img_name = vstrImageLeft[ni].substr(
                            vstrImageLeft[ni].length() - 
                                          KImageNameSuffixLength, 
                                          KImageNameSuffixLength);
       
        
        // Pass the images to the SLAM system
        if (kTrainIntrospectionModel) {
          SLAM.TrackMonocular(imgLeftUnDist,
                           tframe, 
                           cam_poses_gt[ni],
                           img_name,
                           false,
                           qual_img);
        } else {
          SLAM.TrackMonocular(imgLeftUnDist,tframe);
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = 
std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = 
std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> 
>(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory -> currently being saved on shutdown under 
    // SLAM.Shutdown()
//     string path_to_cam_traj;
//     if (kSaveVisualizationsToFile) {
//       path_to_cam_traj = save_visualization_path + "/CameraTrajectory.txt";
//     } else {
//       path_to_cam_traj = "CameraTrajectory.txt";
//     }
//     SLAM.SaveTrajectoryKITTI(path_to_cam_traj);

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/mav0/cam0/data.csv";
    string strPrefixLeft = strPathToSequence + "/mav0/cam0/data/";
    fTimes.open(strPathTimeFile.c_str());
    int count = 0;
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        // Skip the first line
        if(count == 0) continue;
        
        string timestamp_str;
        if(!s.empty()) {
          stringstream ss;
          ss << s;
          getline(ss, timestamp_str, ',');
          double t = stod(timestamp_str) * 1e-9;
          vTimestamps.push_back(t);
        }
        
        vstrImageLeft.push_back(strPrefixLeft + "/" + timestamp_str + ".png");
        
        count++;
    }
}

void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      const string &strPathToPoseUncertainty,
                      const bool &load_pose_uncertainty,
                      vector<string> &vstrImageLeft,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps, 
                      vector<cv::Mat>* cam_pose_gt,
                      vector<Eigen::Vector2f>* rel_cam_pose_uncertainty)
{   
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/mav0/cam0/data.csv";
    string strPrefixLeft = strPathToSequence + "/mav0/cam0/data/";
    vector<long int> img_timestamps_ns;
    vector<long int> gt_timestamps_ns;
    vector<Eigen::Quaternionf> gt_pose_quat_list;
    vector<Eigen::Vector3f> gt_pose_trans_list;
    
    // Vector of the same size as time_stams (image time stamps) with indices 
    // of closest time stamps in gt_time_stamp
    vector<int> matches_img_to_gt;
    vector<bool> frames_synced;
   
    
    // TODO: read calibration from file
    // Hardcoded left camera extrinsics. Transformation from the camera 
    // coordinate frame to body frame
    Eigen::Matrix4f cam0_extrinsics_;
    cam0_extrinsics_ << 
         0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0;
         
    // Transformation from cam0 to the body 
    Eigen::Matrix3f R_C02B = cam0_extrinsics_.topLeftCorner(3,3);
    Eigen::Vector3f t_C02B = cam0_extrinsics_.topRightCorner(3,1);
  
    
    
    fTimes.open(strPathTimeFile.c_str());
    int count = 0;
    while(!fTimes.eof()) {
      count++;
      string s;
      getline(fTimes,s);
      // Skip the first line
      if(count == 1) continue;
      
      string timestamp_str;
      if(!s.empty()) {
        stringstream ss;
        ss << s;
        getline(ss, timestamp_str, ',');
        double t = stod(timestamp_str) * 1e-9;
        vTimestamps.push_back(t);
        img_timestamps_ns.push_back(stol(timestamp_str));
        
        vstrImageLeft.push_back(strPrefixLeft + "/" + timestamp_str + ".png");
      }
    }
    
    // Load ground truth poses
    ifstream fGroundTruthPoses(strPathToGroundTruth.c_str());
    string line;
    
    // Throws away the first line
    std::getline(fGroundTruthPoses, line);
  
    // Read each line
    while (std::getline(fGroundTruthPoses, line)) {
      std::stringstream line_stream(line);
      
      // Parse the values in each line given a comma delimeter
      string word;
      vector<string> parsed_line;
      while(std::getline(line_stream, word, ',')) {
        parsed_line.push_back(word);
      }
     
      gt_timestamps_ns.push_back(stol(parsed_line[0]));
      Eigen::Quaternionf quat(std::stof(parsed_line[4]),
                      std::stof(parsed_line[5]),
                      std::stof(parsed_line[6]),
                      std::stof(parsed_line[7]));
      Eigen::Vector3f trans(std::stof(parsed_line[1]),
                    std::stof(parsed_line[2]),
                    std::stof(parsed_line[3]));
      
      gt_pose_quat_list.push_back(quat);
      gt_pose_trans_list.push_back(trans);
    
      
      if (false) {
        cout << "quat: " << quat.w() << ", " << quat.vec() << endl;
        cout << "trans: " << trans << endl;
      }
    }
    
    // Find the closest frame in the ground truth data to each image timestamp
    SynchronizeTimeStamps(img_timestamps_ns,
                        gt_timestamps_ns,
                        &matches_img_to_gt,
                        &frames_synced);
    
    CHECK_EQ(vstrImageLeft.size(), vTimestamps.size());
    
    vector<int> remove_frames_idx;
    for (size_t i = 0; i < matches_img_to_gt.size(); i++) {
      if (!frames_synced[i]) {
        remove_frames_idx.push_back(i);
        continue;
      }
      
      cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_32F);
      int idx = matches_img_to_gt[i];
      
      Eigen::Matrix3f rot = gt_pose_quat_list[idx].toRotationMatrix();
      
      // EuRoC ground truth poses are in the body reference frame. Convert 
      // them to the left cam coordinate frame.
      Eigen::Matrix3f R_C0_tn_to_t0 = R_C02B.inverse() * rot * R_C02B;
      Eigen::Vector3f trans_C0_tn_to_t0 = 
           -R_C02B.inverse() * t_C02B +
            R_C02B.inverse() * gt_pose_trans_list[idx] +
            R_C02B.inverse() * rot * t_C02B;

      
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          cam_pose.at<float>(i, j) = R_C0_tn_to_t0(i, j);
        }
      }
      
      for (int i = 0; i < 3; i++) {
        cam_pose.at<float>(i, 3) = trans_C0_tn_to_t0(i);
      }
      
      cam_pose_gt->push_back(cam_pose);
    }
    
    // Remove the frames, for which the ground truth pose could not be found
    for (size_t i = 0; i < remove_frames_idx.size(); i++) {
      int idx = *(remove_frames_idx.end() - i - 1);
//       remove_frames_idx.pop_back();
      vstrImageLeft.erase(vstrImageLeft.begin()+idx);
      vTimestamps.erase(vTimestamps.begin()+idx);
    }
   
    
    CHECK_EQ(cam_pose_gt->size(), vTimestamps.size());
   
    // Load reference pose uncertainty values 
    if (load_pose_uncertainty) {
      ifstream fRefPosesUnc;
      fRefPosesUnc.open(strPathToPoseUncertainty.c_str());
      while(!fRefPosesUnc.eof()) {
          string s;
          getline(fRefPosesUnc,s);
          if(!s.empty())
          {
              // pose_uncertainty: (translational_unc, rotational_unc)
              Eigen::Vector2f pose_uncertainty;
              stringstream ss(s);
              string str_curr_entry;
              for (size_t i = 0; i < 2; i++) {
                  getline(ss, str_curr_entry, ' ');
                  pose_uncertainty(i) = stof(str_curr_entry);
              }
              rel_cam_pose_uncertainty->push_back(pose_uncertainty);
          }
      }
      CHECK_EQ(rel_cam_pose_uncertainty->size(), vTimestamps.size());
    }
    
    
    // Load predicted quality images
    int qual_img_num = 0;
    if (kPredictedImageQualityAvailable) {
      if (!GetImageQualFileNames(strPathToImageQual,
                                vTimestamps.size(),
                                &vstrImageQualFilenames,
                                &qual_img_num)){
        LOG(FATAL) << "Error loading the image quality files" << endl;
      }
      LOG(INFO) <<  qual_img_num << " predicted image quality heatmaps found.";
    }
    
    if (kPredictedImageQualityAvailable && qual_img_num < 2) {
      LOG(FATAL) << "Predicted image quality heatmaps not found!";
    }
}


// Given a direcotry whose content is supposed to be files named as ID numbers 
// in the format %010d.jpg, it will fill vstrImageQualFilenames with full path 
// to all available files and leave missing images as empty strings. The 
// argument "size" determines how many images we are expecting starting at
// the index of 0
bool GetImageQualFileNames(const std::string &directory,
                           const int &size,
                           vector<string> *vstrImageQualFilenames,
                           int * num_qual_imgs_found) {
  vstrImageQualFilenames->clear();
  vstrImageQualFilenames->resize(size);
  
  const int kPrefixLength = 10;
  char numbering[10];
  *num_qual_imgs_found = 0;
  
  DIR* dirp = opendir(directory.c_str());
  struct dirent * dp;
  
  if(!dirp) {
    LOG(ERROR) << "Could not open directory " << directory 
               << " for predicted image quality heatmaps.";
  }

  while ((dp = readdir(dirp)) != NULL){

    // Ignore the '.' and ".." directories
    if(!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) continue;
    for(int i = 0; i < kPrefixLength ; i++){
        numbering[i] = dp->d_name[i];
    }
   
    int prefix_number = atoi(numbering);
   
    CHECK_LT(prefix_number, size) << "The index of the images are expected to"
                                  << "be between 0 and " << size - 1;
    vstrImageQualFilenames->at(prefix_number) = directory + "/" +
                                                std::string(dp->d_name);
    *num_qual_imgs_found = *num_qual_imgs_found + 1;
  }
  (void)closedir(dirp);
  
  return true;
}


// Helper function to synchronize two vectors of sorted time stamps (ascending)
// Finds the index of the closest stamp in the second time stamp vector for
// each entry in the first vector
void SynchronizeTimeStamps(const vector<long int>time_stamps_1,
                           const vector<long int>time_stamps_2,
                           vector<int>* matches1to2,
                           vector<bool>* successful_matches) {
  matches1to2->clear();
  successful_matches->clear();
  
  long int max_diff = 1e7;
  
  cout << "time_stamps_1.size(): " << time_stamps_1.size() << endl;
  cout << "time_stamps_2.size(): " << time_stamps_2.size() << endl;
  cout << "time_stamps_1[0]: " << time_stamps_1[0] << endl;
  cout << "time_stamps_2[0]: " << time_stamps_2[0] << endl;
  
  for (size_t i = 0; i < time_stamps_1.size(); i++) {
    
    size_t j = 0;
    // Start the search from the last matched time stamp in time_stamp_2 given
    // the fact that both lists are sorted
    if (i > 0 && successful_matches->at(i - 1)) {
      j = matches1to2->at(i - 1);
    } 
    
    if (time_stamps_1[i] > time_stamps_1[i+1]) {
      LOG(WARNING) << "Non-monotonous time stamp detected!" << std::endl;
    }
    
    bool match_found = false;
    long int last_diff = std::numeric_limits<long int>::max();
   
    while (j < time_stamps_2.size()) {
      long int diff = abs(time_stamps_2[j] - time_stamps_1[i]);
     
      
      if (j < time_stamps_2.size() - 2){
        if (time_stamps_2[j] > time_stamps_2[j+1]) {
          LOG(WARNING) << "Non-monotonous time stamp detected!" << std::endl;
        }
      }
      
      // Throw away datapoints untill the ground truth time stamps overlap with
      // that of the images
      if (time_stamps_1[i] < time_stamps_2[0]) {
        break;
      }
      
      // If the difference starts to increase, the last time stamp has been
      // the closest
      if (diff > last_diff) {
//         double diff_sec = static_cast<double>(diff) * 1e-9;
//         std::cout << "time diff: " << diff_sec << std::endl;
        
        if (diff > max_diff) {
          LOG(FATAL) << "Match found with a time diff " << diff <<
                        "which is larger than the threshold " << max_diff;
        }
        
        match_found = true;
        successful_matches->push_back(true);
        matches1to2->push_back(j - 1);
        break;
      }
      
      last_diff = diff;
      j++;
    }
    
    if (!match_found) {
      successful_matches->push_back(false);
      matches1to2->push_back(0);
      LOG(WARNING) << "Closest ground truth time stamp not found" 
                   << " for image ind " << i;
    }
  }
}
