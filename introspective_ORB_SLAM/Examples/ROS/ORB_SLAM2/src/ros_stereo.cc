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

#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "../../../include/System.h"

// Command line flags flag
DEFINE_string(vocab_path, "", "Path to ORB vocabulary.");

DEFINE_string(settings_path, "", "Path to ORB-SLAM config file.");

DEFINE_string(data_path, "", "Path to the source dataset.");

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
            "to be rectified.");

DEFINE_bool(undistort_images,
            false,
            "Set to true, if input images need "
            "to be undistorted.");

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

  for (const string& arg_name : required_args) {
    bool flag_not_set =
        gflags::GetCommandLineFlagInfoOrDie(arg_name.c_str()).is_default;
    if (flag_not_set) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "stereo_kitti_opt");
      LOG(FATAL) << arg_name << " was not set." << endl;
    }
  }
}

class ImageGrabber {
 public:
  ImageGrabber(ORB_SLAM2::System* pSLAM) : mpSLAM(pSLAM) {}

  void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,
                  const sensor_msgs::ImageConstPtr& msgRight);

  ORB_SLAM2::System* mpSLAM;
  bool do_rectify;
  cv::Mat M1l, M2l, M1r, M2r;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 2;   // ERROR level logging.
  FLAGS_colorlogtostderr = 1;  // Colored logging.
  FLAGS_logtostderr = true;    // Don't log to disk

  ros::init(argc, argv, "stereo");
  ros::start();

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
                         FLAGS_minloglevel,
                         FLAGS_create_ivslam_dataset,
                         FLAGS_run_single_threaded,
                         use_BoW,
                         FLAGS_out_visualization_path,
                         FLAGS_out_dataset_path,
                         silent_mode,
                         guided_ba);

  ImageGrabber igb(&SLAM);

  // Read undistortion/rectification parameters
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

  fsSettings["LEFT.D"] >> D_l;
  fsSettings["RIGHT.D"] >> D_r;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];

  if (FLAGS_rectify_images || FLAGS_undistort_images) {
    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
        R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
      cerr << "ERROR: Calibration parameters to undistort/rectify stereo are "
              "missing!"
           << endl;
      return -1;
    }
  }

  cv::Mat M1l, M2l, M1r, M2r;
  if (FLAGS_rectify_images && FLAGS_undistort_images) {
    cv::initUndistortRectifyMap(K_l,
                                D_l,
                                R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l),
                                CV_32F,
                                igb.M1l,
                                igb.M2l);
    cv::initUndistortRectifyMap(K_r,
                                D_r,
                                R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r),
                                CV_32F,
                                igb.M1r,
                                igb.M2r);
  } else if (FLAGS_rectify_images) {
    D_l = cv::Mat::zeros(1, 4, CV_32F);
    D_r = cv::Mat::zeros(1, 4, CV_32F);

    cv::initUndistortRectifyMap(K_l,
                                D_l,
                                R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l),
                                CV_32F,
                                igb.M1l,
                                igb.M2l);
    cv::initUndistortRectifyMap(K_r,
                                D_r,
                                R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r),
                                CV_32F,
                                igb.M1r,
                                igb.M2r);
  } else if (FLAGS_undistort_images) {
    R_l = cv::Mat::eye(3, 3, CV_32F);
    R_r = cv::Mat::eye(3, 3, CV_32F);
    cv::initUndistortRectifyMap(K_l,
                                D_l,
                                R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l),
                                CV_32F,
                                igb.M1l,
                                igb.M2l);

    cv::initUndistortRectifyMap(K_r,
                                D_r,
                                R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r),
                                CV_32F,
                                igb.M1r,
                                igb.M2r);
  }

  ros::NodeHandle nh;

  message_filters::Subscriber<sensor_msgs::Image> left_sub(
      nh, "/stereo/left/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(
      nh, "/stereo/right/image_raw", 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      sync_pol;
  message_filters::Synchronizer<sync_pol> sync(
      sync_pol(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));

  ros::spin();

  // Stop all threads
  SLAM.Shutdown();

  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
  SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
  SLAM.SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");

  ros::shutdown();

  return 0;
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,
                              const sensor_msgs::ImageConstPtr& msgRight) {
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptrLeft;
  try {
    cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptrRight;
  try {
    cv_ptrRight = cv_bridge::toCvShare(msgRight);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (do_rectify) {
    cv::Mat imLeft, imRight;
    cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
    mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec());
  } else {
    mpSLAM->TrackStereo(cv_ptrLeft->image,
                        cv_ptrRight->image,
                        cv_ptrLeft->header.stamp.toSec());
  }
}
