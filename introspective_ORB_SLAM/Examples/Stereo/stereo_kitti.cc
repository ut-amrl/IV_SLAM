/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
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

#include <System.h>
#include <dirent.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>

#include "MapDrawer.h"
#include "io_access.h"
#include "torch_helpers.h"

#if (CV_VERSION_MAJOR >= 4)
#include <opencv2/imgcodecs/legacy/constants_c.h>
#endif

// Length of the image name suffix that should be extracted from its name
const int KImageNameSuffixLength = 10;

// Global variables
ORB_SLAM2::System *SLAM_ptr;

// Command line flags flag
DEFINE_string(vocab_path, "", "Path to ORB vocabulary.");
DEFINE_string(settings_path, "", "Path to ORB-SLAM config file.");
DEFINE_string(data_path, "", "Path to the source dataset.");
DEFINE_int32(session, -1, "Unique session ID.");
DEFINE_string(ground_truth_path, "", "Path to ground truth poses.");
DEFINE_string(img_qual_path,
              "",
              "Path to quality images for feature "
              "extraction/matching. Higher values of a pixel indicate lower "
              "reliability of the features extracted from that pixel in the "
              "image.");
DEFINE_string(introspection_model_path,
              "",
              "Path to the trained model "
              "of the introspection function.");
DEFINE_string(out_visualization_path,
              "",
              "Output path for visualization "
              "results.");
DEFINE_string(out_dataset_path, "", "Output path for generated dataset.");
DEFINE_string(rel_pose_uncertainty_path,
              "",
              "Path to relative camera pose "
              "uncertainty values.");

DEFINE_int32(start_frame, 0, "Start frame ID.");
DEFINE_int32(end_frame, -1, "End frame ID.");

// If set to true, the estimated camera pose uncertainty values are loaded
// and passed to IV-SLAM
DEFINE_bool(load_rel_pose_uncertainty,
            false,
            "Loads relative camera pose "
            "uncertainty values from file.");

// Set to true if you would like to use predicted heatmaps the same size as
// the input images for weighting the extracted keypoints.
// NOTE: If the program is run in ivslam_enabled  and inference mode but this
// is set to false, it is equivalent to running original ORB-SLAM with the
// additional logging and visualization that is provided in ivslam_enabled mode
DEFINE_bool(load_img_qual_heatmaps,
            false,
            "Loads predicted image quality "
            "heatmpas from file.");

DEFINE_bool(run_single_threaded, false, "Runs in single threaded mode.");
DEFINE_bool(create_ivslam_dataset,
            false,
            "Saves to file the dataset for "
            "training the introspection model.");
DEFINE_bool(ivslam_enabled,
            false,
            "Enables IV-SLAM. The program will run "
            "in trainig mode unless the inference_mode flag is set.");
DEFINE_bool(inference_mode, false, "Enables the inference mode.");
DEFINE_bool(introspection_func_enabled,
            false,
            "Enables the introspection function.");
DEFINE_bool(save_visualizations,
            false,
            "Saves visualization to file if in "
            "ivslam_enabled mode.");
DEFINE_bool(enable_viewer, true, "Enables the viewer.");
DEFINE_bool(gt_pose_available,
            true,
            "If set to true, loads the ground truth "
            "camera poses for either visualizatio or training. This must be "
            "true in training mode or if FLAGS_map_drawer_visualize_gt_pose "
            "is set.");
DEFINE_bool(use_gpu, false, "Uses GPU for running the introspection function.");
DEFINE_bool(rectify_images,
            false,
            "Set to true, if input images need "
            "to be rectified. NOTE: Currently it is assumed that the images "
            "are already undistorted, hence distortion parameters are ignored");

DECLARE_bool(help);
DECLARE_bool(helpshort);

using namespace std;
using namespace ORB_SLAM2;

// Checks if all required command line arguments have been set
void CheckCommandLineArgs(char **argv) {
  vector<string> required_args = {"vocab_path",
                                  "settings_path",
                                  "data_path",
                                  "out_visualization_path",
                                  "out_dataset_path"};

  for (const string &arg_name : required_args) {
    bool flag_not_set =
        gflags::GetCommandLineFlagInfoOrDie(arg_name.c_str()).is_default;
    if (flag_not_set) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "stereo_kitti_opt");
      LOG(FATAL) << arg_name << " was not set." << endl;
    }
  }
}

void LoadImages(const string &strPathToSequence,
                const int &sessionID,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight,
                vector<double> &vTimestamps);

// Loads images as well as the corresponding predicted image quality heatmaps
void LoadImagesWithQual(const string &strPathToSequence,
                        const string &strPathToImageQual,
                        const int &sessionID,
                        vector<string> &vstrImageLeft,
                        vector<string> &vstrImageRight,
                        vector<string> &vstrImageQualFilenames,
                        vector<double> &vTimestamps);

// Loads images as well as the corresponding ground truth camera poses
void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      const string &strPathToPoseUncertainty,
                      const int &sessionID,
                      const bool &load_pose_uncertainty,
                      vector<string> &vstrImageLeft,
                      vector<string> &vstrImageRight,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps,
                      vector<cv::Mat> *cam_pose_gt,
                      vector<Eigen::Vector2f> *rel_cam_pose_uncertainty);

bool GetImageQualFileNames(const std::string &directory,
                           const int &size,
                           vector<string> *vstrImageQualFilenames,
                           int *num_qual_imgs_found);

int GetSmallestImgIdx(const std::string &directory, const int &prefix_length);

void SignalHandler(int signal_num) {
  cout << "Interrupt signal is (" << signal_num << ").\n";

  // terminate program
  if (SLAM_ptr) {
    SLAM_ptr->ShutdownMinimal();
  }

  cout << "Exiting the program!" << endl;

  exit(signal_num);
}

int main(int argc, char **argv) {
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 2;   // ERROR level logging.
  FLAGS_minloglevel = 1;       // WARNING level
  FLAGS_colorlogtostderr = 1;  // Colored logging.
  FLAGS_logtostderr = true;    // Don't log to disk
  signal(SIGINT, SignalHandler);

  string usage(
      "This program runs stereo ORB-SLAM on KITTI format "
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

  torch::jit::script::Module introspection_func;
  torch::Device device = torch::kCPU;
  if (FLAGS_introspection_func_enabled && !FLAGS_load_img_qual_heatmaps) {
    try {
      // Deserialize the ScriptModule from file
      introspection_func = torch::jit::load(FLAGS_introspection_model_path);

      if (FLAGS_use_gpu && torch::cuda::is_available()) {
        std::cout << "Introspection function running on GPU." << std::endl;
        device = torch::kCUDA;
      }
      introspection_func.to(device);
    } catch (const c10::Error &e) {
      std::cerr << "error loading the introspection model\n";
      return -1;
    }
  }

  // Read rectification parameters
  cv::FileStorage fsSettings(FLAGS_settings_path, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
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

  //     fsSettings["LEFT.D"] >> D_l;
  //     fsSettings["RIGHT.D"] >> D_r;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];

  if (FLAGS_rectify_images) {
    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
        R_l.empty() || R_r.empty() || rows_l == 0 || rows_r == 0 ||
        cols_l == 0 || cols_r == 0) {
      cerr << "ERROR: Calibration parameters to rectify stereo are missing!"
           << endl;
      return -1;
    }
  }

  cv::Mat M1l, M2l, M1r, M2r;
  if (FLAGS_rectify_images) {
    cv::initUndistortRectifyMap(K_l,
                                D_l,
                                R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l),
                                CV_32F,
                                M1l,
                                M2l);

    cv::initUndistortRectifyMap(K_r,
                                D_r,
                                R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r),
                                CV_32F,
                                M1r,
                                M2r);
  }

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
    LoadImages(FLAGS_data_path, FLAGS_session, vstrImageLeft, vstrImageRight, vTimestamps);
  } else if (!FLAGS_gt_pose_available) {
    LoadImagesWithQual(FLAGS_data_path,
                       FLAGS_img_qual_path,
                       FLAGS_session, 
                       vstrImageLeft,
                       vstrImageRight,
                       vstrImageQualFilenames,
                       vTimestamps);
  } else {
    LoadImagesWithGT(FLAGS_data_path,
                     FLAGS_ground_truth_path,
                     FLAGS_img_qual_path,
                     FLAGS_rel_pose_uncertainty_path,
                     FLAGS_session, 
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
  if (FLAGS_end_frame > 0) {
    end_frame = std::min(nImages, FLAGS_end_frame);
  } else {
    end_frame = nImages;
  }
  for (int ni = FLAGS_start_frame; ni < end_frame; ni++) {
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 =
        std::chrono::monotonic_clock::now();
#endif
    // Read left and right images from file
    imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_COLOR);
    imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_COLOR);
    // imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
    // imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[ni];

    if (imLeft.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageLeft[ni]) << endl;
      return 1;
    }

    if (imRight.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageRight[ni]) << endl;
      return 1;
    }

    if (FLAGS_rectify_images) {
      cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
      cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);
    } else {
      imLeftRect = imLeft;
      imRightRect = imRight;
    }

    // Read the predicted quality image
    cv::Mat cost_img_cv;
    at::Tensor cost_img;
    if (FLAGS_introspection_func_enabled) {
      if (FLAGS_load_img_qual_heatmaps) {
        // There might not be a image quality available for all input
        // images. In that case just skip the missing ones with setting
        // them to empty images. (The SLAM object will ignore empty
        // images as if no score was available).
        if (vstrImageQualFilenames[ni].empty()) {
          cost_img_cv = cv::Mat(0, 0, CV_8U);
        } else {
          // Read the predicted image quality
          cost_img_cv =
              cv::imread(vstrImageQualFilenames[ni], CV_LOAD_IMAGE_GRAYSCALE);
          if (cost_img_cv.empty()) {
            cerr << endl
                 << "Failed to load image at: " << vstrImageQualFilenames[ni]
                 << endl;
            return 1;
          }
        }
      } else {
        // Run inference on the introspection model online
        cv::Mat imLeft_RGB = imLeft;
        cv::cvtColor(imLeft_RGB, imLeft_RGB, CV_BGR2RGB);

        // Convert to float and normalize image
        imLeft_RGB.convertTo(imLeft_RGB, CV_32FC3, 1.0 / 255.0);
        cv::subtract(imLeft_RGB, cv::Scalar(0.485, 0.456, 0.406), imLeft_RGB);
        cv::divide(imLeft_RGB, cv::Scalar(0.229, 0.224, 0.225), imLeft_RGB);

        auto tensor_img = CVImgToTensor(imLeft_RGB);
        // Swap axis
        tensor_img = TransposeTensor(tensor_img, {(2), (0), (1)});
        // Add batch dim
        tensor_img.unsqueeze_(0);

        tensor_img = tensor_img.to(device);
        std::vector<torch::jit::IValue> inputs{tensor_img};
        cost_img = introspection_func.forward(inputs).toTensor();
        cost_img = (cost_img * 255.0).to(torch::kByte);
        cost_img = cost_img.to(torch::kCPU);

        cost_img_cv = ToCvImage(cost_img);

        // ShowImage(cost_img_cv, "Predicted cost image");
      }

      if (FLAGS_rectify_images) {
        cv::remap(cost_img_cv, cost_img_cv, M1l, M2l, cv::INTER_LINEAR);
      }

      // cv::imshow("heatmap", cost_img_cv);
      // cv::waitKey(0);
    }

    string img_name = vstrImageLeft[ni].substr(
        vstrImageLeft[ni].length() - KImageNameSuffixLength,
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
                         cost_img_cv);
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
                         cost_img_cv);
      }
    } else {
      SLAM.TrackStereo(imLeftRect, imRightRect, tframe);
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 =
        std::chrono::monotonic_clock::now();
#endif

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame. In single threaded mode, load images
    // as fast as possible
    if (!FLAGS_run_single_threaded) {
      double T = 0;
      if (ni < nImages - 1)
        T = vTimestamps[ni + 1] - tframe;
      else if (ni > 0)
        T = tframe - vTimestamps[ni - 1];

      if (ttrack < T) usleep((T - ttrack) * 1e6);
    }
  }

  // Stop all threads
  SLAM.Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  //     cout << "-------" << endl << endl;
  //     cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
  //     cout << "mean tracking time: " << totaltime/nImages << endl;

  cout << "--------------------" << endl;
  cout << "Finished processing sequence located at " << FLAGS_data_path << endl;
  cout << "--------------------" << endl << endl;

  // Save camera trajectory -> currently being saved on shutdown under
  // SLAM.Shutdown()
  //     string path_to_cam_traj;
  //     if (FLAGS_save_visualizations) {
  //       path_to_cam_traj = save_visualization_path + "/CameraTrajectory.txt";
  //     } else {
  //       path_to_cam_traj = "CameraTrajectory.txt";
  //     }
  //     SLAM.SaveTrajectoryKITTI(path_to_cam_traj);

  return 0;
}

void LoadImages(const string &strPathToSequence,
                const int &session,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight,
                vector<double> &vTimestamps) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/" + std::to_string(session) +"_times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }

  string strPrefixLeft = strPathToSequence + "/image_0/";
  string strPrefixRight = strPathToSequence + "/image_1/";

  const int nTimes = vTimestamps.size();
  vstrImageLeft.resize(nTimes);
  vstrImageRight.resize(nTimes);

  int smallest_img_idx = GetSmallestImgIdx(strPrefixLeft, 6);
  for (int i = smallest_img_idx; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i + smallest_img_idx;
    vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
    vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
  }
}

void LoadImagesWithQual(const string &strPathToSequence,
                        const string &strPathToImageQual,
                        const int &session,
                        vector<string> &vstrImageLeft,
                        vector<string> &vstrImageRight,
                        vector<string> &vstrImageQualFilenames,
                        vector<double> &vTimestamps) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/" + std::to_string(session) +"_times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }

  string strPrefixLeft = strPathToSequence + "/image_0/";
  string strPrefixRight = strPathToSequence + "/image_1/";

  const int nTimes = vTimestamps.size();
  vstrImageLeft.resize(nTimes);
  vstrImageRight.resize(nTimes);

  int smallest_img_idx = GetSmallestImgIdx(strPrefixLeft, 6);
  for (int i = smallest_img_idx; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i + smallest_img_idx;
    vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
    vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
  }

  // Load predicted quality images
  int qual_img_num = 0;
  if (FLAGS_load_img_qual_heatmaps && FLAGS_introspection_func_enabled) {
    if (!GetImageQualFileNames(strPathToImageQual,
                               nTimes,
                               &vstrImageQualFilenames,
                               &qual_img_num)) {
      LOG(FATAL) << "Error loading the image quality files" << endl;
    }
    LOG(INFO) << qual_img_num << " predicted image quality heatmaps found.";

    if (qual_img_num != nTimes) {
      LOG(WARNING) << qual_img_num << " predicted image quality heatmaps "
                   << "were found but total session image count is " << nTimes;
    }
    if (qual_img_num < 2) {
      LOG(FATAL) << "Predicted image quality heatmaps not found!";
    }
  }
}

void LoadImagesWithGT(const string &strPathToSequence,
                      const string &strPathToGroundTruth,
                      const string &strPathToImageQual,
                      const string &strPathToPoseUncertainty,
                      const int &session,
                      const bool &load_pose_uncertainty,
                      vector<string> &vstrImageLeft,
                      vector<string> &vstrImageRight,
                      vector<string> &vstrImageQualFilenames,
                      vector<double> &vTimestamps,
                      vector<cv::Mat> *cam_pose_gt,
                      vector<Eigen::Vector2f> *rel_cam_pose_uncertainty) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/" + std::to_string(session) +"_times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }

  string strPrefixLeft = strPathToSequence + "/image_0/";
  string strPrefixRight = strPathToSequence + "/image_1/";

  const int nTimes = vTimestamps.size();
  vstrImageLeft.resize(nTimes);
  vstrImageRight.resize(nTimes);

  int smallest_img_idx = GetSmallestImgIdx(strPrefixLeft, 6);
  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i + smallest_img_idx;
    vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
    vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
  }

  // Load ground truth poses
  ifstream fGroundTruthPoses;
  fGroundTruthPoses.open(strPathToGroundTruth.c_str());
  while (!fGroundTruthPoses.eof()) {
    string s;
    getline(fGroundTruthPoses, s);
    if (!s.empty()) {
      cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_32F);
      stringstream ss(s);
      string str_curr_entry;
      for (size_t i = 0; i < 12; i++) {
        getline(ss, str_curr_entry, ' ');
        cam_pose.at<float>(floor((float)(i) / 4), i % 4) = stof(str_curr_entry);
      }
      cam_pose_gt->push_back(cam_pose);
    }
  }

  CHECK_EQ(cam_pose_gt->size(), nTimes);

  // Load reference pose uncertainty values
  if (load_pose_uncertainty) {
    ifstream fRefPosesUnc;
    fRefPosesUnc.open(strPathToPoseUncertainty.c_str());
    while (!fRefPosesUnc.eof()) {
      string s;
      getline(fRefPosesUnc, s);
      if (!s.empty()) {
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
    CHECK_EQ(rel_cam_pose_uncertainty->size(), nTimes);
  }

  // Load predicted quality images
  int qual_img_num = 0;
  if (FLAGS_load_img_qual_heatmaps && FLAGS_introspection_func_enabled) {
    if (!GetImageQualFileNames(strPathToImageQual,
                               nTimes,
                               &vstrImageQualFilenames,
                               &qual_img_num)) {
      LOG(FATAL) << "Error loading the image quality files" << endl;
    }
    LOG(INFO) << qual_img_num << " predicted image quality heatmaps found.";

    if (qual_img_num != nTimes) {
      LOG(WARNING) << qual_img_num << " predicted image quality heatmaps "
                   << "were found but total session image count is " << nTimes;
    }

    if (qual_img_num < 2) {
      LOG(FATAL) << "Predicted image quality heatmaps not found!";
    }
  }
}

// Given a direcotry whose content is supposed to be files named as ID numbers
// in the format %06d.jpg, it will fill vstrImageQualFilenames with full path
// to all available files and leave missing images as empty strings. The
// argument "size" determines how many images we are expecting starting at
// the index of 0
bool GetImageQualFileNames(const std::string &directory,
                           const int &size,
                           vector<string> *vstrImageQualFilenames,
                           int *num_qual_imgs_found) {
  vstrImageQualFilenames->clear();
  vstrImageQualFilenames->resize(size);

  const int kPrefixLength = 6;
  char numbering[6];
  *num_qual_imgs_found = 0;

  DIR *dirp = opendir(directory.c_str());
  struct dirent *dp;

  if (!dirp) {
    LOG(ERROR) << "Could not open directory " << directory
               << " for predicted image quality heatmaps.";
  }

  while ((dp = readdir(dirp)) != NULL) {
    // Ignore the '.' and ".." directories
    if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) continue;
    for (int i = 0; i < kPrefixLength; i++) {
      numbering[i] = dp->d_name[i];
    }

    int prefix_number = atoi(numbering);

    CHECK_LT(prefix_number, size) << "The index of the images are expected to"
                                  << "be between 0 and " << size - 1;
    vstrImageQualFilenames->at(prefix_number) =
        directory + "/" + std::string(dp->d_name);
    *num_qual_imgs_found = *num_qual_imgs_found + 1;
  }
  (void)closedir(dirp);

  return true;
}

int GetSmallestImgIdx(const std::string &directory, const int &prefix_length) {
  char numbering[20];

  DIR *dirp = opendir(directory.c_str());
  struct dirent *dp;

  if (!dirp) {
    LOG(ERROR) << "Could not open directory " << directory;
  }

  int smallest_idx = std::numeric_limits<int>::max();
  while ((dp = readdir(dirp)) != NULL) {
    // Ignore the '.' and ".." directories
    if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) continue;
    for (int i = 0; i < prefix_length; i++) {
      numbering[i] = dp->d_name[i];
    }

    int prefix_number = atoi(numbering);

    if (prefix_number < smallest_idx) {
      smallest_idx = prefix_number;
    }
  }
  (void)closedir(dirp);

  return smallest_idx;
}
