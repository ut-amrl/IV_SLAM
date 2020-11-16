/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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
#include <gflags/gflags.h>
#include <glog/logging.h>


#include<System.h>
#include "io_access.h"

#if (CV_VERSION_MAJOR >= 4)
  #include<opencv2/imgcodecs/legacy/constants_c.h>
#endif

// Parameters 
// If the stereo images are not already rectified, set this to true
// NOTE: Currently it is assumed that the images are already undistorted, hence
// distortion parameters are ignored
const bool KRectifyImages = true;

// Length of the image name suffix that should be extracted from its name
const int KImageNameSuffixLength = 23;


// Global variables
ORB_SLAM2::System *SLAM_ptr;

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
DEFINE_string(rel_pose_uncertainty_path, "", "Path to relative camera pose "
                                              "uncertainty values.");

DEFINE_int32(start_frame, 0, "Start frame ID.");
DEFINE_int32(end_frame, -1, "End frame ID.");

// If set to ture, the estimated camera pose uncertainty values are loaded 
// and passed to IV-SLAM
DEFINE_bool(load_rel_pose_uncertainty, false, "Loads relative camera pose "
                                           "uncertainty values from file.");

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
DEFINE_bool(gt_pose_available, true, "If set to true, loads the ground truth "
             "camera poses for either visualizatio or training. This must be "
             "true in training mode or if FLAGS_map_drawer_visualize_gt_pose "
             "is set.");

DECLARE_bool(help);
DECLARE_bool(helpshort);

using namespace std;

// Checks if all required command line arguments have been set
void CheckCommandLineArgs(char** argv) {
  vector<string> required_args = {"vocab_path",
                                  "settings_path",
                                  "data_path",
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

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);


// Loads images as well as the corresponding predicted image quality heatmaps
void LoadImagesWithQual(const string &strPathToSequence,
                      const string &strPathToImageQual,
                      vector<string> &vstrImageLeft, 
                      vector<string> &vstrImageRight,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps);

// Loads images as well as the corresponding ground truth camera poses
void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      const string &strPathToPoseUncertainty,
                      const bool &load_pose_uncertainty,
                      vector<string> &vstrImageLeft, 
                      vector<string> &vstrImageRight,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps, 
                      vector<cv::Mat>* cam_pose_gt,
                      vector<Eigen::Vector2f>* rel_cam_pose_uncertainty);

bool GetImageQualFileNames(const std::string &directory,
                           const int &size,
                           const vector<string>& vstrLeftImgNames,
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
    
    string usage("This program runs stereo ORB-SLAM on EuROC format "
                 "data with the option to run with IV-SLAM in inference mode "
                 "or generate training data for it. \n");
    
    usage += string(argv[0]) + " <argument1> <argument2> ...";
    gflags::SetUsageMessage(usage);
    
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "stereo_kitti");
      return 0;
    }
    CheckCommandLineArgs(argv);
   
    if (!FLAGS_gt_pose_available && FLAGS_ivslam_enabled &&   
        !FLAGS_inference_mode) {
      LOG(FATAL) << "Ground truth camera poses are required in training mode.";
    }

    if (!FLAGS_gt_pose_available && FLAGS_map_drawer_visualize_gt_pose) {
      LOG(FATAL) << "Ground truth camera poses are not available but their "
                 << "visualization is requested!";
    }
    
    
    // Read rectification parameters
    cv::FileStorage fsSettings(FLAGS_settings_path, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() 
          || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" 
             << endl;
        return -1;
    }

    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,
                            P_l.rowRange(0,3).colRange(0,3),
                            cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    
    cv::initUndistortRectifyMap(K_r,D_r,R_r,
                            P_r.rowRange(0,3).colRange(0,3),
                            cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);

    
    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrDepthImageFilenames;
    vector<string> vstrImageQualFilenames;
    vector<double> vTimestamps;
    vector<cv::Mat> cam_poses_gt;
    vector<Eigen::Vector2f> rel_cam_poses_uncertainty;
    // The map from left image names to the ID of corresponding relative camera 
    // pose uncertainty (from current image to next image)
    std::unordered_map<std::string, int> pose_unc_map;
   
   
    if (!FLAGS_ivslam_enabled) {
      LoadImages(FLAGS_data_path, vstrImageLeft, vstrImageRight, vTimestamps);
    } else if (!FLAGS_gt_pose_available) {
      LoadImagesWithQual(FLAGS_data_path,
                        FLAGS_img_qual_path,
                        vstrImageLeft, 
                        vstrImageRight,
                        vstrImageQualFilenames,
                        vTimestamps);
    } else {
      LoadImagesWithGT(FLAGS_data_path, 
                       FLAGS_ground_truth_path,
                       FLAGS_img_qual_path,
                       FLAGS_rel_pose_uncertainty_path,
                       FLAGS_load_rel_pose_uncertainty,
                       vstrImageLeft, 
                       vstrImageRight,
                       vstrImageQualFilenames,
                       vTimestamps,
                       &cam_poses_gt,
                       &rel_cam_poses_uncertainty);
      
      if (FLAGS_load_rel_pose_uncertainty) {
        cout << "loading rel pose unc " << endl;
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
    bool use_BoW = true;
    bool silent_mode = false;
    bool guided_ba = false;
    ORB_SLAM2::System SLAM(FLAGS_vocab_path,
                           FLAGS_settings_path,
                           ORB_SLAM2::System::STEREO,
                           FLAGS_enable_viewer,
                           FLAGS_ivslam_enabled,
                           FLAGS_inference_mode,
                           FLAGS_save_visualizations,
                           FLAGS_create_ivslam_dataset,
                           FLAGS_run_single_threaded,
                           use_BoW,
                           FLAGS_out_visualization_path,
                           FLAGS_out_dataset_path,
                           silent_mode,
                           guided_ba);
    if (FLAGS_load_rel_pose_uncertainty) {
      SLAM.SetRelativeCamPoseUncertainty(&pose_unc_map, 
                                        &rel_cam_poses_uncertainty);
    }
  
    SLAM_ptr = &SLAM;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   
    cout << "Start frame: " << FLAGS_start_frame << endl;

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    int end_frame;
    if(FLAGS_end_frame > 0) {
      end_frame = std::min(nImages, FLAGS_end_frame);
    } else {
      end_frame = nImages;
    }
    for(int ni = FLAGS_start_frame; ni < end_frame; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }
        
        if(imRight.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageRight[ni]) << endl;
            return 1;
        }
        
        if (KRectifyImages) {
          cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
          cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
        } else {
          imLeftRect = imLeft;
          imRightRect = imRight;
        }
        
        // Read the predicted quality image
        cv::Mat qual_img;
        if (FLAGS_load_img_qual_heatmaps) {
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

            if (KRectifyImages) {
              cv::remap(qual_img,qual_img,M1l,M2l,cv::INTER_LINEAR);
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
        
        Eigen::Matrix<double, 6, 6> cam_poses_gt_cov;
        bool pose_cov_available = false;
        
        // Pass the images to the SLAM system
        if (FLAGS_ivslam_enabled) {
          if (FLAGS_gt_pose_available) {
            SLAM.TrackStereo(imLeftRect,
                            imRightRect,
                            tframe, 
                            cam_poses_gt[ni],
                            cam_poses_gt_cov,
                            pose_cov_available,
                            img_name,
                            false,
                            qual_img);
          } else {
            cv::Mat cam_pose_gt = cv::Mat(0, 0, CV_32F);
            SLAM.TrackStereo(imLeftRect,
                imRightRect,
                tframe, 
                cam_pose_gt,
                cam_poses_gt_cov,
                pose_cov_available,
                img_name,
                false,
                qual_img);
          }
        } else {
          SLAM.TrackStereo(imLeftRect,imRightRect,tframe);
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

        // Wait to load the next frame. In single threaded mode, load images
        // as fast as possible          
        if(!FLAGS_run_single_threaded) {
          double T=0;
          if(ni<nImages-1)
              T = vTimestamps[ni+1]-tframe;
          else if(ni>0)
              T = tframe-vTimestamps[ni-1];

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
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/mav0/cam0/data.csv";
    string strPrefixLeft = strPathToSequence + "/mav0/cam0/data/";
    string strPrefixRight = strPathToSequence + "/mav0/cam1/data/";
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
        vstrImageRight.push_back(strPrefixRight + "/" + timestamp_str + ".png");
        
        count++;
    }
}

void LoadImagesWithQual(const string &strPathToSequence,
                      const string &strPathToImageQual,
                      vector<string> &vstrImageLeft, 
                      vector<string> &vstrImageRight,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps) {
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/mav0/cam0/data.csv";
    string strPrefixLeft = strPathToSequence + "/mav0/cam0/data/";
    string strPrefixRight = strPathToSequence + "/mav0/cam1/data/";
    vector<long int> img_timestamps_ns;
    vector<long int> gt_timestamps_ns;

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
        vstrImageRight.push_back(strPrefixRight + "/" + timestamp_str + ".png");
      }
    }

    // Load predicted quality images
    int qual_img_num = 0;
    if (FLAGS_load_img_qual_heatmaps) {
      if (!GetImageQualFileNames(strPathToImageQual,
                                vTimestamps.size(),
                                vstrImageLeft,
                                &vstrImageQualFilenames,
                                &qual_img_num)){
        LOG(FATAL) << "Error loading the image quality files" << endl;
      }
      LOG(INFO) <<  qual_img_num << " predicted image quality heatmaps found.";
    }
    
    if (FLAGS_load_img_qual_heatmaps && qual_img_num < 2) {
      LOG(FATAL) << "Predicted image quality heatmaps not found!";
    }
    
}

void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      const string &strPathToPoseUncertainty,
                      const bool &load_pose_uncertainty,
                      vector<string> &vstrImageLeft, 
                      vector<string> &vstrImageRight,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps, 
                      vector<cv::Mat>* cam_pose_gt,
                      vector<Eigen::Vector2f>* rel_cam_pose_uncertainty)
{   
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/mav0/cam0/data.csv";
    string strPrefixLeft = strPathToSequence + "/mav0/cam0/data/";
    string strPrefixRight = strPathToSequence + "/mav0/cam1/data/";
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
        vstrImageRight.push_back(strPrefixRight + "/" + timestamp_str + ".png");
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
    CHECK_EQ(vstrImageRight.size(), vTimestamps.size());
    
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
      vstrImageRight.erase(vstrImageRight.begin()+idx);
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
    if (FLAGS_load_img_qual_heatmaps) {
      if (!GetImageQualFileNames(strPathToImageQual,
                                vTimestamps.size(),
                                vstrImageLeft,
                                &vstrImageQualFilenames,
                                &qual_img_num)){
        LOG(FATAL) << "Error loading the image quality files" << endl;
      }
      LOG(INFO) <<  qual_img_num << " predicted image quality heatmaps found.";
    }
    
    if (FLAGS_load_img_qual_heatmaps && qual_img_num < 2) {
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
                           const vector<string>& vstrLeftImgNames,
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

  for (size_t i = 0; i < vstrLeftImgNames.size(); i++) {
    string img_name = vstrLeftImgNames[i];
    // string tmp_str = img_name.substr(0, img_name.size()-4);
    // vstrImageQualFilenames->at(i) = tmp_str + .jpg";

    vstrImageQualFilenames->at(i) = directory + "/" 
                                    +img_name.substr(img_name.size()-23, 
                                                     23-4) 
                                     + ".jpg";

    *num_qual_imgs_found = *num_qual_imgs_found + 1;
  }

  // while ((dp = readdir(dirp)) != NULL){

  //   // Ignore the '.' and ".." directories
  //   if(!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) continue;
  //   for(int i = 0; i < kPrefixLength ; i++){
  //       numbering[i] = dp->d_name[i];
  //   }
   
  //   int prefix_number = atoi(numbering);
   
  //   CHECK_LT(prefix_number, size) << "The index of the images are expected to"
  //                                 << "be between 0 and " << size - 1;
  //   vstrImageQualFilenames->at(prefix_number) = directory + "/" +
  //                                               std::string(dp->d_name);
  //   *num_qual_imgs_found = *num_qual_imgs_found + 1;
  // }
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
