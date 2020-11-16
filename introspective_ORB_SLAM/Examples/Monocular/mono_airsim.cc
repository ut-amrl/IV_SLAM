// Copyright 2019 srabiee@cs.utexas.edu
// Department of Computer Science,
// University of Texas at Austin
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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry> 
#include<opencv2/core/core.hpp>
#include <random>
#include <dirent.h>
#include <csignal> 
#include <gflags/gflags.h>
#include <glog/logging.h>

#include"System.h"
#include "io_access.h"

#if (CV_VERSION_MAJOR >= 4)
  #include<opencv2/imgcodecs/legacy/constants_c.h>
#endif

// Parameters 

const bool kApplyNoiseToCamPose = false;
// If set to true noise is added to the pose of camera based on a noise model
// (motion model)
const bool kUseNoiseModel = false;

// If set to true, the camera pose covariance will be passed to ORB-SLAM
const bool kPassPoseCov = true;

// float kAngualrVariance = 0.5 * M_PI / 180.0; // radians : 2deg
// float kTranslationalVariance = 0.02; // meters : 0.1m
float kAngualrVariance = 0.0005; // radians : 2deg
float kTranslationalVariance = 0.0005; // meters : 0.1m

// If set to true, the replay rate of images will be upper bounded by the 
// provided threshold instead of doing so using the frames' timestamps. 
// This is only for the multithreaded mode as in the single-threaded mode 
// images will be processed as fast as possible.
const bool kEnforceReplayRate = true;

const float kReplayRateMaxThresh = 30.0; // Hz 

// Length of the image name suffix that should be extracted from its name
const int KImageNameSuffixLength = 14;

// Global variables
ORB_SLAM2::System *SLAM_ptr;

std::default_random_engine rnd_generator;


// Command line flags flag
DEFINE_string(vocab_path, "", "Path to ORB vocabulary.");
DEFINE_string(settings_path, "", "Path to ORB-SLAM config file.");
DEFINE_string(data_path, "", "Path to the source dataset.");
DEFINE_string(ground_truth_path, "", "Path to ground truth poses.");
DEFINE_string(img_qual_path, "", "Path to quality images for feature "
             "extraction/matching. Higher values of a pixel indicate lower "
             "reliability of the features extracted from that pixel in the "
             "image.");
DEFINE_string(out_visualization_path, "", "Output path for visualization "
                                          "results.");
DEFINE_string(out_dataset_path, "", "Output path for generated dataset.");

DEFINE_int32(start_frame, 0, "Start frame ID.");
DEFINE_int32(end_frame, -1, "End frame ID.");

DEFINE_bool(load_gt_depth_imgs, false, "Loads the ground truth depth images "
                  "and uses them for image feature evaluation if in "
                  "training mode. ");

// Set to true if you would like to use predicted heatmaps the same size as
// the input images for weighting the extracted keypoints.
// NOTE: If the program is run in ivslam_enabled  and inference mode but this 
// is set to false, it is equivalent to running original ORB-SLAM with the 
// additional logging and visualization that is provided in ivslam_enabled mode
DEFINE_bool(load_img_qual_heatmaps, false, "Loads predicted image quality "
                                           "heatmpas from file.");

DEFINE_bool(run_single_threaded, false, "Runs in single threaded mode.");
DEFINE_bool(create_ivslam_dataset, false, "Saves to file the dataset for "
                                          "training the introspection model.");
DEFINE_bool(ivslam_enabled, false, "Enables IV-SLAM. The program will run "
               "in trainig mode unless the inference_mode flag is set.");
DEFINE_bool(inference_mode, false, "Enables the inference mode.");
DEFINE_bool(save_visualizations, false, "Saves visualization to file if in "
                                      "ivslam_enabled mode.");
DEFINE_bool(enable_viewer, true, "Enables the viewer.");

DECLARE_bool(help);
DECLARE_bool(helpshort);


using namespace ORB_SLAM2;
using namespace std;
using Eigen::Vector3f;
using Eigen::Matrix3f;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::AngleAxisd;

// Checks if all required command line arguments have been set
void CheckCommandLineArgs(char** argv) {
  vector<string> required_args = {"vocab_path",
                                  "settings_path",
                                  "data_path",
                                  "ground_truth_path",
                                  "out_visualization_path",
                                  "out_dataset_path"};
  
  for (const string& arg_name:required_args) {
    bool flag_not_set =   
          gflags::GetCommandLineFlagInfoOrDie(arg_name.c_str()).is_default;
    if (flag_not_set) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "stereo_kitti_opt");
      LOG(FATAL) << arg_name <<  " was not set." << endl;
    }
  }
}

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

// Loads images as well as the corresponding ground truth camera poses
// void LoadImagesWithGT(const string &strPathToSequence,
//                       const string &strPathToGroundTruth,
//                       const string &strPathToImageQual,
//                       const Matrix3d &rot_base_leftcamStd,
//                       const Vector3d &t_base_leftcamStd,
//                       vector<string> &vstrImageFilenames,
//                       vector<string> &vstrDepthImageFilenames,
//                       vector<string> &vstrImageQualFilenames,
//                       vector<double> &vTimestamps, 
//                       vector<cv::Mat>* cam_pose_gt);

void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      vector<string> &vstrImageLeft,
                      vector<string> &vstrDepthImageFilenames,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps, 
                      vector<cv::Mat>* cam_pose_gt,
                      vector<Eigen::Matrix<float, 6, 6>> *cam_pose_gt_cov);

bool GetImageQualFileNames(const std::string &directory,
                           const int &size,
                           vector<string> *vstrImageQualFilenames,
                           int * num_qual_imgs_found);

// Given the pose of camera at previous frame and current frame, it returns
// a covariance matrix associated to the pose of the camera at current frame.
// The covariance is estimated using noise model that is dependent on the 
// motion of the camera in between the two frames.
void ApplyNoiseModel(const Eigen::Matrix3d &rot_prev,
                     const Eigen::Vector3d &trans_prev,
                     const Eigen::Matrix3d &rot_curr,
                     const Eigen::Vector3d &trans_curr,
                     Eigen::Matrix3d *rot_curr_noisy,
                     Eigen::Vector3d *trans_curr_noisy,
                     Eigen::Matrix<float, 6, 6> *cov_curr);

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
  
    string usage("This program runs monocular ORB-SLAM on airsim format "
                 "data with the option to run with IV-SLAM in inference mode "
                 "or generate training data for it. \n");
    
    usage += string(argv[0]) + " <argument1> <argument2> ...";
    gflags::SetUsageMessage(usage);
    
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "mono_airsim");
      return 0;
    }
    CheckCommandLineArgs(argv);
    
    if(FLAGS_load_gt_depth_imgs && FLAGS_load_img_qual_heatmaps) {
      LOG(FATAL) << "Only one of FLAGS_load_gt_depth_imgs and "
                "FLAGS_load_img_qual_heatmaps can be set at a time." << endl;
    }

    CHECK_EQ(kUseNoiseModel, false) << "Variable camera pose covariance "
                         "from a noise model is not supported in monocular "
                         "mode yet.";

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrDepthImageFilenames;
    vector<string> vstrImageQualFilenames;
    vector<double> vTimestamps;
    vector<cv::Mat> cam_poses_gt;
    vector<Eigen::Matrix<float, 6, 6>> cam_poses_gt_cov;
    
    if (!FLAGS_ivslam_enabled) {
      LoadImages(FLAGS_data_path, vstrImageFilenames, vTimestamps);
    } else {
      LoadImagesWithGT(FLAGS_data_path, 
                       FLAGS_ground_truth_path,
                       FLAGS_img_qual_path,
                       vstrImageFilenames,
                       vstrDepthImageFilenames,
                       vstrImageQualFilenames,
                       vTimestamps,
                       &cam_poses_gt,
                       &cam_poses_gt_cov);
    }
  
    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to 
    // process frames.
    bool use_BoW = true;
    bool silent_mode = false;
    ORB_SLAM2::System SLAM(FLAGS_vocab_path,
                           FLAGS_settings_path,
                           ORB_SLAM2::System::MONOCULAR,
                           FLAGS_enable_viewer,
                           FLAGS_ivslam_enabled,
                           FLAGS_inference_mode,
                           FLAGS_save_visualizations,
                           FLAGS_create_ivslam_dataset,
                           FLAGS_run_single_threaded,
                           use_BoW,
                           FLAGS_out_visualization_path,
                           FLAGS_out_dataset_path,
                           silent_mode);
    
    SLAM_ptr = &SLAM;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    cout << "Start frame: " << FLAGS_start_frame << endl;

    // Main loop
    cv::Mat im;
    
    int end_frame;
    if(FLAGS_end_frame > 0) {
      end_frame = std::min(nImages, FLAGS_end_frame);
    } else {
      end_frame = nImages;
    }
    for(int ni=FLAGS_start_frame; ni<end_frame; ni++)
    {
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = 
                            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = 
                            std::chrono::monotonic_clock::now();
#endif
        double tframe = vTimestamps[ni];
        
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        if(im.empty()) {
            cerr << endl << "Failed to load image at: " << 
                            vstrImageFilenames[ni] << endl;
            return 1;
        }
        
        // Read the depth image
        cv::Mat depth_im;
        if (FLAGS_load_gt_depth_imgs) {
          PFM pfm_rw;
          float * depth_data = 
                        pfm_rw.read_pfm<float>(vstrDepthImageFilenames[ni]);
          depth_im = cv::Mat(pfm_rw.getHeight(), 
                                    pfm_rw.getWidth(), 
                                    CV_32F, 
                                    depth_data);
          cv::flip(depth_im, depth_im, 0);
          
          // Visualize and verify the depth image
  //         double min;
  //         double max;
  //         cv::minMaxIdx(depth_im, &min, &max);
  //         cv::Mat adjMap;
  //         cv::convertScaleAbs(depth_im, adjMap, 255.0 / max);
  //         cv::imshow("window", adjMap); 
  //         cv::waitKey(0);
          
          if(depth_im.empty()) {
            cerr << endl << "Failed to load depth image at: " << 
                            vstrDepthImageFilenames[ni] << endl;
            return 1;
          }
        } else if (FLAGS_load_img_qual_heatmaps) {
          // There might not be a image quality available for all input 
          // images. In that case just skip the missing ones with setting 
          // them to empty images. (The SLAM object will ignore empty
          // images as if no score was available).
          if(vstrImageQualFilenames[ni].empty()) {
            depth_im = cv::Mat(0, 0, CV_8U);
          } else {
          
            // Read the predicted image quality
            depth_im = cv::imread(
                            vstrImageQualFilenames[ni],CV_LOAD_IMAGE_GRAYSCALE);
            if(depth_im.empty()) {
                cerr << endl << "Failed to load image at: " << 
                                vstrImageQualFilenames[ni] << endl;
                return 1;
            }
          }
        }
       
        string img_name = vstrImageFilenames[ni].substr(
                              vstrImageFilenames[ni].length() - 
                              KImageNameSuffixLength, KImageNameSuffixLength);
               
        // Pass the image to the SLAM system.
        // "depth_im" is either the ground truth depth image or the predicted
        // image quality given the input parameters.
        if (FLAGS_ivslam_enabled) {
          SLAM.TrackMonocular(im,tframe, 
                              cam_poses_gt[ni], 
                              img_name, 
                              FLAGS_load_gt_depth_imgs,
                              depth_im);
        } else {
          SLAM.TrackMonocular(im,tframe);
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = 
                                  std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = 
                                  std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<
                          std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame. In single threaded mode, load images
        // as fast as possible
        if(!FLAGS_run_single_threaded) {
          double T=0;
          if(!kEnforceReplayRate) {
            if(ni<nImages-1)
                T = vTimestamps[ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestamps[ni-1];
          } else {
            T = 1.0 / kReplayRateMaxThresh;
          }

          if(ttrack<T)
              usleep((T-ttrack)*1e6);
        }
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
//     cout << "-------" << endl << endl;
//     cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
//     cout << "mean tracking time: " << totaltime/nImages << endl;

    cout << "--------------------" << endl;
    cout << "Finished processing sequence located at " 
         << FLAGS_data_path << endl;
    cout << "--------------------" << endl << endl;


    // Save camera trajectory
//     string path_to_cam_traj;
//     if (kSaveVisualizationsToFile) {
//       path_to_cam_traj = save_visualization_path+ "/KeyFrameTrajectory_TUM.txt";
//     } else {
//       path_to_cam_traj = "KeyFrameTrajectory_TUM.txt";
//     }
//     SLAM.SaveTrajectoryKITTI(path_to_cam_traj);
//     SLAM.SaveKeyFrameTrajectoryTUM(path_to_cam_traj);    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> 
&vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times_img.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t * 1e-9);
        }
    }

    string strPrefixLeft = strPathToSequence + "/img_left/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}


void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      vector<string> &vstrImageLeft,
                      vector<string> &vstrDepthImageFilenames,
                      vector<string> &vstrImageQualFilenames,
                    vector<double> &vTimestamps, 
                    vector<cv::Mat>* cam_pose_gt,
                    vector<Eigen::Matrix<float, 6, 6>> *cam_pose_gt_cov) {
    std::default_random_engine generator;
    std::normal_distribution<double> rot_dist(0.0,
                                      sqrt(kAngualrVariance));
    std::normal_distribution<double> trans_dist(0.0,
                                      sqrt(kTranslationalVariance));
    
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times_img.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t * 1e-9);
          }
    }

    string strPrefixLeft = strPathToSequence + "/img_left/";
    string strPrefixRight = strPathToSequence + "/img_right/";
    string strPrefixDepth = strPathToSequence + "/img_depth/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrDepthImageFilenames.resize(nTimes);
    vstrImageQualFilenames.resize(nTimes);

    
    for(int i=0; i<nTimes; i++)
    {
        stringstream ss_10sig;
        ss_10sig << setfill('0') << setw(10) << i;
        vstrImageLeft[i] = strPrefixLeft + ss_10sig.str() + ".png";
        vstrDepthImageFilenames[i] = strPrefixDepth + ss_10sig.str() + ".pfm";
    }
    
    int qual_img_num = 0;
    if (FLAGS_load_img_qual_heatmaps) {
      if (!GetImageQualFileNames(strPathToImageQual,
                                nTimes,
                                &vstrImageQualFilenames,
                                &qual_img_num)){
        LOG(FATAL) << "Error loading the image quality files" << endl;
      }
      LOG(INFO) <<  qual_img_num << " predicted image quality heatmaps found.";
      
      if(qual_img_num != nTimes) {
        LOG(WARNING) << qual_img_num << " predicted image quality heatmaps "
                     << "were found but total session image count is "
                     << nTimes;
      }
    }
    
    if (FLAGS_load_img_qual_heatmaps && qual_img_num < 2) {
      LOG(FATAL) << "Predicted image quality heatmaps not found!";
    }
   
   
    // Load ground truth poses
    ifstream fGroundTruthPoses;
    fGroundTruthPoses.open(strPathToGroundTruth.c_str());
    
    // The pose of previous frame
    Eigen::Matrix3d rot_world_leftcamStdPrev = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_world_leftcamStdPrev = Eigen::Vector3d::Zero();
   
    int line_count = 0;
    
    // Read the Ground truth left cam poses from file
    while(!fGroundTruthPoses.eof()) {
        string s;
        getline(fGroundTruthPoses,s);
        if(!s.empty())
        {
            cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_32F);
            Eigen::Matrix<float, 6, 6> cam_pose_cov;
            stringstream ss(s);
            string str_curr_entry;
            vector<string> line_entry;
            for (size_t i = 0; i < 8; i++) {
                getline(ss, str_curr_entry, ' ');
                line_entry.push_back(str_curr_entry);
            }
            
            // Transform the car pose from quaternion to rotation matrix
            Eigen::Quaterniond quat(stod(line_entry[7]),
                                    stod(line_entry[4]),
                                    stod(line_entry[5]),
                                    stod(line_entry[6]));
            Eigen::Vector3d t_world_leftcamStd(stod(line_entry[1]),
                                  stod(line_entry[2]),
                                  stod(line_entry[3]));
            Eigen::Matrix3d rot_world_leftcamStd = quat.toRotationMatrix();
            
            if (line_count == 0) {
              rot_world_leftcamStdPrev = rot_world_leftcamStd;
              t_world_leftcamStdPrev = t_world_leftcamStd;
            }
                   
            // Add Noise to the camera pose
            if (kApplyNoiseToCamPose) {
              if (kUseNoiseModel) {
                Eigen::Vector3d t_world_leftcamStd_noisy;
                Eigen::Matrix3d rot_world_leftcamStd_noisy;
                ApplyNoiseModel(rot_world_leftcamStdPrev,
                     t_world_leftcamStdPrev,
                     rot_world_leftcamStd,
                     t_world_leftcamStd,
                     &rot_world_leftcamStd_noisy,
                     &t_world_leftcamStd_noisy,
                     &cam_pose_cov);
                rot_world_leftcamStd = rot_world_leftcamStd_noisy;
                t_world_leftcamStd = t_world_leftcamStd_noisy;
              } else {
                // Add Noise from a constant distribution to the pose of camera
                // in all frames.
                // Sample noise from a Gaussian distribution given the variance
                double del_w1 = rot_dist(generator);
                double del_w2 = rot_dist(generator);
                double del_w3 = rot_dist(generator);
                double del_x = trans_dist(generator);
                double del_y = trans_dist(generator);
                double del_z = trans_dist(generator);
                
                Eigen::Vector3d w(del_w1, del_w2, del_w3);
                // Create the rotation matrix for the sampled perturbation 
                // vector w (the lie algebra representation) 
                Eigen::Matrix3d rot_perturbation; 
                rot_perturbation = Eigen::AngleAxisd(w.norm(), w / w.norm());
                
                // Perturb the rotational part of camera pose
                rot_world_leftcamStd = 
                                rot_perturbation * rot_world_leftcamStd;
              
                // Perturb the translational part of camera pose
                t_world_leftcamStd = t_world_leftcamStd + Vector3d(del_x,
                                                                  del_y,
                                                                  del_z);
               
                // Fill the covariance matrix
                cam_pose_cov <<
                  kAngualrVariance, 0, 0, 0, 0, 0,
                  0, kAngualrVariance, 0, 0, 0, 0,
                  0, 0, kAngualrVariance, 0, 0, 0,
                  0, 0, 0, kTranslationalVariance, 0, 0,
                  0, 0, 0, 0, kTranslationalVariance, 0,
                  0, 0, 0, 0, 0, kTranslationalVariance;
              }
            }
                        
            
            // Convert the eigen rotation and tranlation to cv mat
            for (size_t i = 0; i < 3; i++) {
              for (size_t j = 0; j < 3; j++) {
                cam_pose.at<float>(i, j) = 
                      static_cast<float>(rot_world_leftcamStd(i, j)); 
              }
            }
            
            for (size_t i = 0; i < 3; i++) {
              cam_pose.at<float>(i, 3) = t_world_leftcamStd(i);
            }
            cam_pose_gt->push_back(cam_pose);
            cam_pose_gt_cov->push_back(cam_pose_cov);
            
            rot_world_leftcamStdPrev = rot_world_leftcamStd;
            t_world_leftcamStdPrev = t_world_leftcamStd;
            line_count++;
        }
    }
    
    CHECK_EQ(cam_pose_gt->size(), nTimes);
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

// Given the pose of camera at previous frame and current frame, it returns
// a covariance matrix associated to the pose of the camera at current frame.
// The covariance is estimated using noise model that is dependent on the 
// motion of the camera in between the two frames.
void ApplyNoiseModel(const Eigen::Matrix3d &rot_prev,
                     const Eigen::Vector3d &trans_prev,
                     const Eigen::Matrix3d &rot_curr,
                     const Eigen::Vector3d &trans_curr,
                     Eigen::Matrix3d *rot_curr_noisy,
                     Eigen::Vector3d *trans_curr_noisy,
                     Eigen::Matrix<float, 6, 6> *cov_curr) {
  // Parameters of the noise model
//   const double K1 = 5e-4 * 1;
//   const double K2 = 5e-4 * 1;
//   const double K3 = 1e-2 * 1;
//   const double K4 = 1e-2 * 1;
  const double K1 = 5e-4 * 1;
  const double K2 = 5e-4 * 1;
  const double K3 = 1e-2 * 1;
  const double K4 = 1e-2 * 1;
  
//   const double w_bias = 1e-10;
//   const double t_bias = 1e-10;
  const double w_bias = 0;
  const double t_bias = 0;
  
 
  // The standard normal distribution (u=0.0, sigma=1.0)
  std::normal_distribution<float> std_normal(0.0, 1.0);
  
  // Compute the motion of camera in between current frame and previous frame 
  // Takes points from current frame to previous frame
  Eigen::Matrix3d rot_prev_curr = rot_prev.transpose() * rot_curr;
  Eigen::Vector3d trans_prev_curr = -rot_prev.transpose() * trans_prev +
                                    rot_prev.transpose() * trans_curr;
  
  // The angle-axis representation of rot_prev_curr
  Eigen::AngleAxisd aa_prev_curr;
  aa_prev_curr = rot_prev_curr;
  double rot_ang = aa_prev_curr.angle();
  double trans_mag = trans_prev_curr.norm();
  
//   // TODO: remove this temp testing
//   double del_rot = kAngualrVariance;
  double del_rot = w_bias + K1 * rot_ang + K2 * trans_mag;
  // The covariance matrix of the rotaional part of tf_prev_curr (w12)
  Eigen::Matrix3d sigma_rot = Eigen::Matrix3d::Identity() * del_rot;
  
//   // TODO: remove this temp testing
//   Eigen::Vector3d del_t (kTranslationalVariance,
//                          kTranslationalVariance,
//                          kTranslationalVariance);
  Eigen::Vector3d del_t = Eigen::Vector3d::Ones() * t_bias +
                          Eigen::Vector3d::Ones() * rot_ang * K3 +
                          trans_prev_curr * K4;
//   Eigen::Vector3d del_t = Eigen::Vector3d::Ones() * t_bias +
//                           Eigen::Vector3d::Ones() * rot_ang * K3 +
//                           Eigen::Vector3d::Ones() * trans_mag * K4;
  
  // The covariance matrix of the translational part of tf_prev_curr (t12)
  Eigen::Matrix3d sigma_trans = del_t.asDiagonal();
  
  // The jacobian of the rotational part of t_curr w.r.t. the rotational
  // part of tf_prev_curr (w12)
  Eigen::Matrix3d J_w12 = rot_prev;
  
  // The jacobian of the translational part of t_curr w.r.t. the rotational
  // part of tf_prev_curr (t12)
  Eigen::Matrix3d J_t12 = rot_prev;
  
  // The covariance of the rotational part of tf_curr (w2)
  Eigen::Matrix3d sigma_w2 = J_w12 * sigma_rot * J_w12.transpose();
  
  // The covariance of the translational part of tf_curr (t2)
  Eigen::Matrix3d sigma_t2 = J_t12 * sigma_trans * J_t12.transpose();
  
  cov_curr->setZero();
  cov_curr->topLeftCorner(3, 3) = sigma_w2.cast<float>();
  cov_curr->bottomRightCorner(3, 3) = sigma_t2.cast<float>();
  
  // Sample a transformation from a multivariate normal distribution with the 
  // covariance matrix of cov_curr. Use the transformation to perturb the pose
  // of current frame
  
  // Compute the Cholesky decomposition of cov_curr, i.e. find L such that
  // L * LT = cov_curr
  Eigen::Matrix<float, 6, 6> L;
  L = cov_curr->llt().matrixL();
  
  // Sample from the standard normal distribution and then multiply by L to 
  // keep the covariance of the distribution be cov_curr
  Eigen::Matrix<float, 6, 1> delta_transform;
  for (size_t i = 0; i < 6; i++) {
    float sample = std_normal(rnd_generator);
    delta_transform(i) = sample;
  }
  delta_transform = L * delta_transform;
  
  // Rotational part of delta_transform
  Eigen::Vector3d delta_w = delta_transform.topLeftCorner(3,1).cast<double>();
  
  // Translational part of delta_transform
  Eigen::Vector3d delta_t = 
                  delta_transform.bottomLeftCorner(3,1).cast<double>();
  
  Eigen::Matrix3d rot_perturbation;
  if (delta_w.norm() <= std::numeric_limits<double>::min()) {
    rot_perturbation = Eigen::Matrix3d::Identity();
  } else {
    rot_perturbation = Eigen::AngleAxisd(delta_w.norm(), 
                                       delta_w / delta_w.norm());
  }        
  
  // Perturb the rotational part of camera pose
  *rot_curr_noisy = rot_perturbation * rot_curr;
            
  // Perturb the translational part of camera pose
  *trans_curr_noisy = trans_curr + delta_t;
 
  
//   cout << "delta_transform" << delta_transform.transpose() << endl;
//   cout << *cov_curr << endl;
//   cout << L * L.transpose() << endl;
//   cout << "original trans: " << trans_curr.transpose() << endl;
//   cout << "noisy trnas: " << trans_curr_noisy->transpose() << endl;
//   cout << endl;
}
