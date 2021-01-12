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

#include "feature_evaluator.h"

namespace feature_evaluation {

using Eigen::AngleAxisf;
using Eigen::Matrix3d;
using Eigen::Matrix3f;
using Eigen::MatrixXf;
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::VectorXf;
using std::cout;
using std::endl;
using std::mutex;
using std::string;
using std::unique_lock;
using std::vector;
using namespace cv;
// using namespace cv::xfeatures2d;
using namespace ORB_SLAM2;

FeatureEvaluator::FeatureEvaluator(DescriptorType descriptor_type,
                                   Dataset dataset)
    : descriptor_type_(descriptor_type), dataset_(dataset) {
  if (training_mode_ == kCompareAgainstPrevFrameEpipolar ||
      training_mode_ == kCompareAgainstRefKeyFrameEpipolar ||
      training_mode_ == kCompareAgainstPrevFrameEpipolarNormalized ||
      training_mode_ == kCompareAgainstRefKeyFrameEpipolarNormalized) {
    error_type_ = kEpipolar;
  } else {
    error_type_ = kReprojection;
  }

  if (error_type_ == kEpipolar) {
    kBadFeatureErrThresh_ = kBadFeatureErrThresh_Epipolar_;

    // Use 0.7 when doing no error normalization (1.0 for high)
    // Use 0.005 when doing normalization using constant perturbations
    kErrHeatmapMaxVal_ = 0.9 * kBadFeatureErrThresh_Epipolar_;  // 0.2, 0.7

  } else {
    kBadFeatureErrThresh_ = kBadFeatureErrThresh_Reproj_;
    kErrHeatmapMaxVal_ = 2 * kBadFeatureErrThresh_Reproj_;
  }
}

void FeatureEvaluator::LoadImagePair(cv::Mat img_prev, cv::Mat img_curr) {
  // Images of the Euroc dataset need to be undistorted
  if (dataset_ == kEuroc) {
    if (!camera_calib_loaded_) {
      LOG(FATAL) << "Camera calibration was not loaded.";
    }
    undistort(img_prev, img_prev_, camera_mat_, dist_coeffs_cv_);
    undistort(img_curr, img_curr_, camera_mat_, dist_coeffs_cv_);
  } else {
    img_prev_ = img_prev;
    img_curr_ = img_curr;
  }
}

void FeatureEvaluator::EvaluateFeatures(ORB_SLAM2::Frame& prev_frame,
                                        ORB_SLAM2::Frame& curr_frame) {
  unique_lock<mutex> lock(MapPoint::mGlobalMutex);
  // TODO (srabiee): uncomment this and scale the ground truth
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, prev_frame.mTwc_gt);
  // TODO (srabiee): Read scale in a systematic manner
  // Fix scale for KITTI 07
  //   float scale = 6.4327805070470; // KITTI 07
  //   float scale = 10.413217; // KITTI 08
  float scale = 1.0;
  tf_prev_to_curr.rowRange(0, 3).col(3) =
      tf_prev_to_curr.rowRange(0, 3).col(3) / scale;
  //   cv::Mat tf_prev_to_curr = CalculateRelativeTransform(
  //                     CalculateInverseTransform(curr_frame.mTcw),
  //                     CalculateInverseTransform(prev_frame.mTcw));

  keypts2_select_.clear();
  keypts2_select_life_.clear();
  keypts2_select_ref_frame_offset_.clear();
  reproj2_select_.clear();
  reproj2_select_gt_.clear();
  epipolar_lines_dir_.clear();
  epipolar_projections_.clear();
  sigma_pts_err_.clear();
  err_norm_factor_.clear();
  epi_err_vec_particles_cov_.clear();
  keypts2_matched_w_prev_.clear();
  keypts1_matched_w_curr_.clear();
  matches1to2_.clear();
  bad_keypts1_matched_w_curr_.clear();
  bad_keypts2_matched_w_prev_.clear();
  bad_matches1to2_.clear();
  err_vals_select_.clear();
  err_vals_visualization_.clear();

  switch (training_mode_) {
    case kCompareAgainstPrevFrame:
      EvaluateAgainstPrevFrame(prev_frame, curr_frame);
      break;
    case kCompareAgainstPrevFrameAndLastKeyFrame:
      LOG(FATAL) << "This training mode is not implemented!";
      EvaluateAgainstPrevFrameAndLastKeyFrame(prev_frame, curr_frame);
      break;
    case kCompareAgainstPrevFrameAndRefKeyFrame:
      EvaluateAgainstPrevFrameAndRefKeyFrame(prev_frame, curr_frame);
      break;
    case kCompareAgainstLastKeyFrame:
      LOG(FATAL) << "This training mode is not implemented!";
      EvaluateAgainstLastKeyFrame(prev_frame, curr_frame);
      break;
    case kCompareAgainstRefKeyFrame:
      EvaluateAgainstRefKeyFrame(prev_frame, curr_frame);
      break;
    case kCompareAgainstPrevFrameEpipolar:
      EvaluateAgainstPrevFrameEpipolar(prev_frame, curr_frame);
      break;
    case kCompareAgainstRefKeyFrameEpipolar:
      EvaluateAgainstRefKeyFrameEpipolar(prev_frame, curr_frame);
      break;
    case kCompareAgainstPrevFrameEpipolarNormalized:
      EvaluateAgainstPrevFrameEpipolarNormalized(prev_frame, curr_frame);
      break;
    case kCompareAgainstRefKeyFrameEpipolarNormalized:
      EvaluateAgainstRefKeyFrameEpipolarNormalized(prev_frame, curr_frame);
      break;
    default:
      LOG(FATAL) << "Unknown training mode!";
  }

  // Calculate the statistics of the epipolar error values
  //   cout << "error size: " << err_vals_select_.size() << endl;
  //   if (err_vals_select_.size() > 0) {
  //     auto it_max = max_element(std::begin(err_vals_select_),
  //                           std::end(err_vals_select_));
  //     auto it_min = min_element(std::begin(err_vals_select_),
  //                           std::end(err_vals_select_));
  //     cout << "max error: " << *it_max << endl;
  //     cout << "min error: " << *it_min << endl;
  //     cout << endl;
  //   }

  // Use log(epipolar_err) to better differentiate small differences
  // Also count the percentage of feature matches that are bad (this is
  // all matches including the ones to reference frame and not just those
  // from previous frame used for visualization)
  // Error values are categorized into good or bad and binary values are
  // assigned to them if kMakeErrValuesBinary_ is set to true.
  unsigned int bad_match_count_all = 0;
  if (err_vals_select_.size() > 0) {
    for (size_t i = 0; i < err_vals_select_.size(); i++) {
      //       err_vals_select_[i] = log(err_vals_select_[i]);
      if (err_vals_select_[i] > kBadFeatureErrThresh_) {
        bad_match_count_all++;
        if (kMakeErrValuesBinary_) {
          err_vals_select_[i] = kBadFeatureErrVal_;
        }
      } else if (kMakeErrValuesBinary_) {
        err_vals_select_[i] = kGoodFeatureErrVal_;
      }
    }

    bad_matches_percent_ = static_cast<float>(bad_match_count_all) * 100.0 /
                           static_cast<float>(err_vals_select_.size());

    // Statistics of the epipolar error normalization factor values
    //   if (!err_norm_factor_.empty()) {
    //     double sum = std::accumulate(err_norm_factor_.begin(),
    //                                  err_norm_factor_.end(), 0);
    //     double mean = sum / static_cast<double>(err_norm_factor_.size());
    //     auto it_max_f = max_element(std::begin(err_norm_factor_),
    //                           std::end(err_norm_factor_));
    //     auto it_min_f = min_element(std::begin(err_norm_factor_),
    //                           std::end(err_norm_factor_));
    //
    //     cout << "err_norm_factor_mean: " <<  mean << endl;
    //     cout << "err_norm_fac_max: " << *it_max_f << endl;
    //     cout << "err_norm_fac_min: " << *it_min_f <<  endl;
    //   }

    // Statistics of the epipolar error variance values
    //   if (!epi_err_vec_particles_cov_.empty()) {
    //     vector<float> epi_err_sigma(epi_err_vec_particles_cov_.size());
    //     for (size_t i = 0; i < epi_err_sigma.size(); i++) {
    //       epi_err_sigma[i] = sqrt(epi_err_vec_particles_cov_[i](0,0));
    //     }
    //
    //     double sum = std::accumulate(epi_err_sigma.begin(),
    //                                  epi_err_sigma.end(), 0);
    //     double mean = sum / static_cast<double>(epi_err_sigma.size());
    //     auto it_max_f = max_element(std::begin(epi_err_sigma),
    //                           std::end(epi_err_sigma));
    //     auto it_min_f = min_element(std::begin(epi_err_sigma),
    //                           std::end(epi_err_sigma));
    //
    //     cout << "ep_err_sigma_mean: " <<  mean << endl;
    //     cout << "ep_err_95conf_mean: " <<  mean * sqrt(5.991) << endl;
    //     cout << "ep_err_sigma_max: " << *it_max_f << endl;
    //     cout << "ep_err_sigma_min: " << *it_min_f <<  endl;
    //   }

  } else {
    bad_matches_percent_ = 0;
  }

  // Find bad matches for visualization purposes
  int bad_match_counter = 0;
  for (size_t i = 0; i < keypts2_matched_w_prev_.size(); i++) {
    if (err_vals_visualization_[i] > kBadFeatureErrThresh_) {
      bad_keypts1_matched_w_curr_.push_back(keypts1_matched_w_curr_[i]);
      bad_keypts2_matched_w_prev_.push_back(keypts2_matched_w_prev_[i]);
      bad_matches1to2_.push_back(
          DMatch(bad_match_counter, bad_match_counter, 0.0));
      bad_match_counter++;
    }
  }

  // Generate a feature matching image
  //   drawMatches(img_prev_, keypts1_matched_w_curr_, img_curr_,
  //               keypts2_matched_w_prev_, matches1to2_,
  //               img_matching_annotation_);

  // Generate a feature matching image only for the bad features
  //   drawMatches(img_prev_, bad_keypts1_matched_w_curr_, img_curr_,
  //                bad_keypts2_matched_w_prev_,
  //               bad_matches1to2_, img_bad_matching_annotation_);
}

void FeatureEvaluator::UpdateCameraCalibration(const ORB_SLAM2::Frame& frame) {
  projection_mat_cam0_ << frame.fx, 0, frame.cx, 0, 0, frame.fy, frame.cy, 0, 0,
      0, 1.0, 0;

  projection_mat_cam1_ << frame.fx, 0, frame.cx, -frame.mb, 0, frame.fy,
      frame.cy, 0, 0, 0, 1.0, 0;

  camera_mat_ = (Mat_<double>(3, 3) << frame.fx,
                 0,
                 frame.cx,
                 0,
                 frame.fy,
                 frame.cy,
                 0,
                 0,
                 1.0);

  dist_coeffs_cv_ = (Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0);

  camera_calib_loaded_ = true;
}

void FeatureEvaluator::LoadRectificationMap(const std::string& calib_file) {
  cv::FileStorage file(calib_file, cv::FileStorage::READ);
  if (!file.isOpened()) {
    LOG(FATAL) << "ERROR: Wrong path to calibration file: " << calib_file;
  }
  // TODO: Find a way to align this with the output of the stereo calibration format
  cv::Mat K_l, P_l, R_l, D_l;
  file["LEFT.K"] >> K_l;
  file["LEFT.P"] >> P_l;
  file["LEFT.R"] >> R_l;
  file["LEFT.D"] >> D_l;

  int rows_l = file["LEFT.height"];
  int cols_l = file["LEFT.width"];

  if (K_l.empty() || P_l.empty() || R_l.empty() || rows_l == 0 || cols_l == 0) {
    LOG(WARNING) << "Required parameters for image rectification were not "
                 << "found in the calibration file. Generated training "
                 << "heatmaps will not be unrectified.";
    rectification_map_available_ = false;
    return;
  }

  // The unrectification is currently only supported for undistorted images
  if (!D_l.empty()) {
    if (cv::countNonZero(D_l) > 0) {
      LOG(FATAL) << "Heatmap unrectification is not currently supported for "
                 << "distorted images.";
    }
  }

  cv::Mat R_l_inv;
  cv::transpose(R_l, R_l_inv);
  cv::initUndistortRectifyMap(P_l.rowRange(0, 3).colRange(0, 3),
                              D_l,
                              R_l_inv,
                              K_l,
                              cv::Size(cols_l, rows_l),
                              CV_32F,
                              unrect_map1_left_,
                              unrect_map2_left_);
  rectification_map_available_ = true;
}

std::vector<cv::KeyPoint> FeatureEvaluator::GetMatchedKeyPoints() {
  return keypts2_select_;
}

cv::Mat FeatureEvaluator::GetAnnotatedImg() {
  if (dataset_ == kEuroc) {
    return img_matching_annotation_;
  } else if (dataset_ == kKITTI) {
    cv::Mat scaled_img;
    cv::resize(img_matching_annotation_, scaled_img, cv::Size(), 0.7, 0.7);
    return scaled_img;
  } else {
    LOG(FATAL) << "Unknown dataset!";
  }
}

cv::Mat FeatureEvaluator::GetBadMatchAnnotatedImg() {
  if (dataset_ == kEuroc) {
    return img_bad_matching_annotation_;
  } else if (dataset_ == kKITTI) {
    cv::Mat scaled_img;
    cv::resize(img_bad_matching_annotation_, scaled_img, cv::Size(), 0.7, 0.7);
    return scaled_img;
  } else {
    LOG(FATAL) << "Unknown dataset!";
  }
}

cv::Mat FeatureEvaluator::GetUndistortedImg() { return img_prev_; }

cv::Mat FeatureEvaluator::GetFeatureErrVisualization() {
  const bool auto_color_scaling = false;
  const int radius = 5;  // 5, 3
                         //   const float err_min_clamp = 0.0;
                         //   const float err_max_clamp = kErrHeatmapMaxVal_;
  const float err_min_clamp = kErrMinClamp_;
  const float err_max_clamp = kErrMaxClamp_;

  //   img_curr_.copyTo(img_feature_qual_annotation_);
  cvtColor(img_curr_, img_feature_qual_annotation_, COLOR_GRAY2BGR);
  vector<cv::KeyPoint>* keypts_of_interest;
  vector<float>* reproj_err_of_interest;
  switch (visualization_mode_) {
    case kVisualizeAll: {
      keypts_of_interest = &keypts2_select_;
      reproj_err_of_interest = &err_vals_select_;
      break;
    }
    case kVisualizeOnlyMatchedWithRef: {
      keypts_of_interest = &keypts2_matched_w_prev_;
      reproj_err_of_interest = &err_vals_visualization_;
      break;
    }
    default:
      LOG(FATAL) << "Unknown visualization mode!";
  }

  vector<float>& err_vals = *reproj_err_of_interest;
  for (size_t i = 0; i < keypts_of_interest->size(); i++) {
    Point point(static_cast<int>(keypts_of_interest->at(i).pt.x),
                static_cast<int>(keypts_of_interest->at(i).pt.y));
    //     Scalar green = Scalar(0, 255, 0);
    int scaled_err = 0;
    if (auto_color_scaling) {
      // Find the max and min epipolar error values
      double err_min = *min_element(std::begin(err_vals), std::end(err_vals));
      double err_max = *max_element(std::begin(err_vals), std::end(err_vals));
      scaled_err = static_cast<int>(255.0 * (err_vals[i] - err_min) /
                                    (err_max - err_min));
      //       cout << "err max/min: " << err_max << ", " << err_min << endl;
      //       cout << "err, scaled_err:  " << err_vals[i] << ", " << scaled_err
      //          << endl;
    } else {
      scaled_err = static_cast<int>(255.0 * (err_vals[i] - err_min_clamp) /
                                    (err_max_clamp - err_min_clamp));
      scaled_err = (scaled_err > 255) ? 255 : scaled_err;
      scaled_err = (scaled_err < 0) ? 0 : scaled_err;
    }
    Scalar color = Scalar(125, scaled_err, 0);
    circle(img_feature_qual_annotation_, point, radius, color, -1, 8, 0);
    //     cout << "err_vals: " << err_vals[i] << endl;
  }
  return img_feature_qual_annotation_;
}

// A generelized version of GetFeatureErrVisualization that visualizes
// a vector of keypoints on current frame and colors them given a scalar value
// provided for each keypoint (the scalcar could be the value of corresponding
// error for that keypoint for instance)
cv::Mat FeatureEvaluator::ColorKeypoints(cv::Mat& img,
                                         const vector<cv::KeyPoint>& keypts,
                                         const vector<double>& scalars,
                                         double scalar_max_clamp,
                                         double drawing_radius) {
  const bool auto_color_scaling = false;
  const int radius = drawing_radius;  // 5, 3
  const float err_min_clamp = 0.0;
  const float err_max_clamp = scalar_max_clamp;

  CHECK_EQ(keypts.size(), scalars.size());

  //   img_curr_.copyTo(img);
  cvtColor(img_curr_, img, COLOR_GRAY2BGR);

  for (size_t i = 0; i < keypts.size(); i++) {
    Point point(static_cast<int>(keypts[i].pt.x),
                static_cast<int>(keypts[i].pt.y));
    //     Scalar green = Scalar(0, 255, 0);
    int scaled_err = 0;
    if (auto_color_scaling) {
      // Find the max and min epipolar error values
      double err_min = *min_element(std::begin(scalars), std::end(scalars));
      double err_max = *max_element(std::begin(scalars), std::end(scalars));
      scaled_err = static_cast<int>(255.0 * (scalars[i] - err_min) /
                                    (err_max - err_min));
      //       cout << "err max/min: " << err_max << ", " << err_min << endl;
      //       cout << "err, scaled_err:  " << err_vals[i] << ", " << scaled_err
      //          << endl;
    } else {
      scaled_err = static_cast<int>(255.0 * (scalars[i] - err_min_clamp) /
                                    (err_max_clamp - err_min_clamp));
    }
    Scalar color = Scalar(125, scaled_err, 0);
    circle(img, point, radius, color, -1, 8, 0);
    //     cout << "scalars: " << scalars[i] << endl;
  }
  return img;
}

cv::Mat FeatureEvaluator::GetBadRegionHeatmap() {
  if (rectification_map_available_) {
    return UnrectifyImage(bad_region_heatmap_);
  } else {
    return bad_region_heatmap_;
  }
}

cv::Mat FeatureEvaluator::GetBadRegionHeatmapMask() {
  if (rectification_map_available_) {
    return UnrectifyImage(bad_region_heatmap_mask_);
  } else {
    return bad_region_heatmap_mask_;
  }
}

std::vector<float> FeatureEvaluator::GetErrValues() { return err_vals_select_; }

void FeatureEvaluator::GenerateImageQualityHeatmap() {
  vector<double> x_in(err_vals_select_.size());
  vector<double> y_in(err_vals_select_.size());
  for (size_t i = 0; i < err_vals_select_.size(); i++) {
    x_in[i] = keypts2_select_[i].pt.x;
    y_in[i] = keypts2_select_[i].pt.y;
  }

  // The remaining strip at the right and bottom of the image are cropped out
  // This should be taken into accout during training
  int bin_num_x =
      std::floor((double(img_curr_.cols) - kBinSizeX_) / kBinStride_) + 1;
  int bin_num_y =
      std::floor((double(img_curr_.rows) - kBinSizeY_) / kBinStride_) + 1;

  // freq holds the sum of the weights of all features in each bin
  Eigen::ArrayXXd freq(bin_num_x, bin_num_y);
  Eigen::ArrayXXd bin_val(bin_num_x, bin_num_y);
  freq.setZero(bin_num_x, bin_num_y);
  //   bin_val.setZero(bin_num_x, bin_num_y);
  bin_val.setOnes(bin_num_x, bin_num_y);
  bin_val *= kErrMinClamp_;

  Hist2D(x_in,
         y_in,
         err_vals_select_,
         kBinSizeX_,
         kBinSizeY_,
         kBinStride_,
         kBinStride_,
         bin_num_x,
         bin_num_y,
         &freq,
         &bin_val);

  // Combines the frequency and mean error value information to generate
  // image quality scores
  Eigen::ArrayXXd bin_mean_val(bin_val);
  Eigen::ArrayXXd bin_score(bin_num_x, bin_num_y);
  //   Eigen::ArrayXXd freq_clamped = ClampArray(freq, kMaxBinFreq_);

  // Calculate the mean error value for each bin
  for (int i = 0; i < freq.size(); i++) {
    if (freq(i) > 0) {
      bin_mean_val(i) = bin_val(i) / static_cast<double>(freq(i));
    }
  }

  // Calculate the score metric that you want to use for the training
  bin_score = bin_mean_val;
  //   bin_score = log(bin_mean_val);
  //   bin_score = 1.0 * log(bin_mean_val) + 1.0 * (1/(1 +
  //   freq.cast<double>()));

  // Create a heatmap image of the bad regions for SLAM/VO
  vector<double> bin_score_vec;
  bin_score_vec.assign(bin_score.data(), bin_score.data() + bin_score.size());
  cv::Mat bad_region_heatmap_low_res =
      GenerateErrHeatmap(bin_num_y, bin_num_x, bin_score_vec);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  bad_region_heatmap_.release();
  cv::resize(bad_region_heatmap_low_res,
             bad_region_heatmap_,
             cv::Size((bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                      (bin_num_y - 1) * kBinStride_ + kBinSizeY_));

  bad_region_heatmap_.convertTo(bad_region_heatmap_, CV_8U, 255.0);
  //   cv::Mat heatmap_overlaid = OverlayHeatmapOnImage(bad_region_heatmap_);

  //   cout << "epp_err size: " << reproj_err_.size() << endl;
  //   cout << "freq.size(): " << freq.size() << endl;
  //   cout << "bin_num_x: " << bin_num_x << endl;
  //   cout << "bin_num_y: " << bin_num_y << endl;
  //   cout << "freq array: " << endl << freq << endl;
  //   cout << "clamped freq array: " << endl << freq_clamped << endl;
  //   cout << "bin_mean_val: " << bin_mean_val << endl;
  //   cout << "bin_score: " << bin_score << endl;
  //   cout << "bad_region_heatmap_low_res: " << endl
  //        << bad_region_heatmap_low_res << endl;
}

void FeatureEvaluator::GenerateUnsupImageQualityHeatmap(
    ORB_SLAM2::Frame& frame, const std::string target_path) {
  int N = frame.N;
  vector<int> idx_interest;
  idx_interest.reserve(N);

  for (size_t i = 0; i < N; i++) {
    if (frame.mvChi2Dof[i] > 0) {
      idx_interest.push_back(i);
    }
  }

  vector<float> err_vals_vec(idx_interest.size());
  vector<double> x_in(idx_interest.size());
  vector<double> y_in(idx_interest.size());

  for (size_t i = 0; i < idx_interest.size(); i++) {
    err_vals_vec[i] =
        (2 / (1 + frame.mvKeyQualScoreTrain[idx_interest[i]])) - 1;
    x_in[i] = static_cast<double>(frame.mvKeysUn[idx_interest[i]].pt.x);
    y_in[i] = static_cast<double>(frame.mvKeysUn[idx_interest[i]].pt.y);
  }

  // The remaining strip at the right and bottom of the image are cropped out
  // This should be taken into accout during training
  int bin_num_x =
      std::floor((double(img_curr_.cols) - kBinSizeX_) / kBinStride_) + 1;
  int bin_num_y =
      std::floor((double(img_curr_.rows) - kBinSizeY_) / kBinStride_) + 1;

  // freq holds the sum of the weights of all features in each bin
  Eigen::ArrayXXd freq(bin_num_x, bin_num_y);
  Eigen::ArrayXXd bin_val(bin_num_x, bin_num_y);
  freq.setZero(bin_num_x, bin_num_y);
  //   bin_val.setZero(bin_num_x, bin_num_y);
  bin_val.setOnes(bin_num_x, bin_num_y);
  bin_val *= kErrMinClamp_;

  Hist2D(x_in,
         y_in,
         err_vals_vec,
         kBinSizeX_,
         kBinSizeY_,
         kBinStride_,
         kBinStride_,
         bin_num_x,
         bin_num_y,
         &freq,
         &bin_val);

  // Combines the frequency and mean error value information to generate
  // image quality scores
  Eigen::ArrayXXd bin_mean_val(bin_val);
  Eigen::ArrayXXd bin_score(bin_num_x, bin_num_y);
  //   Eigen::ArrayXXd freq_clamped = ClampArray(freq, kMaxBinFreq_);

  // Calculate the mean error value for each bin
  for (int i = 0; i < freq.size(); i++) {
    if (freq(i) > 0) {
      bin_mean_val(i) = bin_val(i) / static_cast<double>(freq(i));
    }
  }

  // Calculate the score metric that you want to use for the training
  bin_score = bin_mean_val;
  //   bin_score = log(bin_mean_val);
  //   bin_score = 1.0 * log(bin_mean_val) + 1.0 * (1/(1 +
  //   freq.cast<double>()));

  // Create a heatmap image of the bad regions for SLAM/VO
  vector<double> bin_score_vec;
  bin_score_vec.assign(bin_score.data(), bin_score.data() + bin_score.size());
  cv::Mat bad_region_heatmap_low_res =
      GenerateErrHeatmap(bin_num_y, bin_num_x, bin_score_vec, 1.0, 0.0);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  cv::Mat bad_region_heatmap;
  cv::resize(bad_region_heatmap_low_res,
             bad_region_heatmap,
             cv::Size((bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                      (bin_num_y - 1) * kBinStride_ + kBinSizeY_));

  bad_region_heatmap.convertTo(bad_region_heatmap, CV_8U, 255.0);

  // Overlay the heatmap on the image
  cv::Mat heatmap_colored;
  cv::applyColorMap(bad_region_heatmap, heatmap_colored, cv::COLORMAP_JET);

  // TODO: pass current image as an argument
  cv::Mat left_img_color;
  cvtColor(img_curr_, left_img_color, cv::COLOR_GRAY2BGR);

  cv::Mat overlaid_heatmap;
  addWeighted(left_img_color, 0.5, heatmap_colored, 0.5, 0.0, overlaid_heatmap);

  cv::imshow("bad_region_heatmap", overlaid_heatmap);
  cv::waitKey(0);

  string output_dir = target_path + "/bad_region_heatmap_unsupervised_vis/";
  CreateDirectory(target_path);
  CreateDirectory(output_dir);
  cv::imwrite(output_dir + frame.mstrLeftImgName, overlaid_heatmap);
}

void FeatureEvaluator::GenerateImageQualityHeatmapGP() {
  // The remaining strip at the right and bottom of the image are cropped out
  // This should be taken into accout during training
  int bin_num_x =
      std::floor((double(img_curr_.cols) - kBinSizeX_) / kBinStride_) + 1;
  int bin_num_y =
      std::floor((double(img_curr_.rows) - kBinSizeY_) / kBinStride_) + 1;

  if (err_vals_select_.empty()) {
    LOG(INFO) << "No keypoints available for heatmap generation!";
    bad_region_heatmap_ = cv::Mat((bin_num_y - 1) * kBinStride_ + kBinSizeY_,
                                  (bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                                  CV_8U);
    return;
  }

  vector<Vector2f> point_loc(err_vals_select_.size());
  for (size_t i = 0; i < err_vals_select_.size(); i++) {
    point_loc[i] = Vector2f(keypts2_select_[i].pt.x, keypts2_select_[i].pt.y);
  }
  MatrixXf kmat = Kmatrix(point_loc);
  Eigen::Map<VectorXf> err_vals(err_vals_select_.data(),
                                err_vals_select_.size());

  vector<double> grid_quality_vec(bin_num_x * bin_num_y);

  // Predict/interpolate the error value for each of the points on a grid
  for (int j = 0; j < bin_num_y; j++) {
    for (int i = 0; i < bin_num_x; i++) {
      float x = i * kBinStride_ + kBinSizeX_ / 2.0;
      float y = j * kBinStride_ + kBinSizeY_ / 2.0;
      float mean;
      float variance;
      GPPredict(x, y, point_loc, err_vals, kmat, mean, variance);
      grid_quality_vec[i + j * bin_num_x] = static_cast<double>(mean);
    }
  }

  // Create a heatmap image of the bad regions for SLAM/VO
  cv::Mat bad_region_heatmap_low_res =
      GenerateErrHeatmap(bin_num_y, bin_num_x, grid_quality_vec);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  bad_region_heatmap_.release();
  cv::resize(bad_region_heatmap_low_res,
             bad_region_heatmap_,
             cv::Size((bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                      (bin_num_y - 1) * kBinStride_ + kBinSizeY_));

  bad_region_heatmap_.convertTo(bad_region_heatmap_, CV_8U, 255.0);
}

void FeatureEvaluator::GenerateUnsupImageQualityHeatmapGP(
    ORB_SLAM2::Frame& frame) {
  // The threshold that is used to generate a binary mask from the
  // Gaussian Process estimated variance values that are normalized btw 0 and 1
  const float kNormalizedGPVarThresh = 0.5;

  // The maximum threshold used for normalizing the estimated variance values
  // by the Gaussian Process.
  const float kGPVarMaxThresh = 100.0;  // 200.0

  int N = frame.N;
  vector<int> idx_interest;
  idx_interest.reserve(N);

  for (size_t i = 0; i < N; i++) {
    if (frame.mvChi2Dof[i] > 0) {
      idx_interest.push_back(i);
    }
  }

  vector<float> err_vals_vec(idx_interest.size());
  vector<Vector2f> point_loc(idx_interest.size());

  for (size_t i = 0; i < idx_interest.size(); i++) {
    err_vals_vec[i] =
        (2 / (1 + frame.mvKeyQualScoreTrain[idx_interest[i]])) - 1;
    point_loc[i] = Vector2f(frame.mvKeysUn[idx_interest[i]].pt.x,
                            frame.mvKeysUn[idx_interest[i]].pt.y);
  }

  // The remaining strip at the right and bottom of the image are cropped out
  // This should be taken into accout during training
  int bin_num_x =
      std::floor((double(img_curr_.cols) - kBinSizeX_) / kBinStride_) + 1;
  int bin_num_y =
      std::floor((double(img_curr_.rows) - kBinSizeY_) / kBinStride_) + 1;

  if (err_vals_vec.empty()) {
    LOG(INFO) << "No keypoints available for heatmap generation!";
    bad_region_heatmap_ = cv::Mat((bin_num_y - 1) * kBinStride_ + kBinSizeY_,
                                  (bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                                  CV_8U);
    return;
  }

  MatrixXf kmat = Kmatrix(point_loc);
  Eigen::Map<VectorXf> err_vals(err_vals_vec.data(), err_vals_vec.size());

  vector<double> grid_quality_vec(bin_num_x * bin_num_y);
  vector<double> grid_qual_var_vec(bin_num_x * bin_num_y);

  // Predict/interpolate the error value for each of the points on a grid
  for (int j = 0; j < bin_num_y; j++) {
    for (int i = 0; i < bin_num_x; i++) {
      float x = i * kBinStride_ + kBinSizeX_ / 2.0;
      float y = j * kBinStride_ + kBinSizeY_ / 2.0;
      float mean;
      float variance;
      GPPredict(x, y, point_loc, err_vals, kmat, mean, variance);
      grid_quality_vec[i + j * bin_num_x] = static_cast<double>(mean);
      grid_qual_var_vec[i + j * bin_num_x] = static_cast<double>(variance);
    }
  }

  // +++++++++++++++
  // Use the GP variance values to generate a reliability mask for the
  // image quality heatmap
  cv::Mat bad_region_var_heatmap_low_res = GenerateErrHeatmap(
      bin_num_y, bin_num_x, grid_qual_var_vec, kGPVarMaxThresh, 0.0);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  cv::resize(bad_region_var_heatmap_low_res,
             bad_region_heatmap_mask_,
             cv::Size((bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                      (bin_num_y - 1) * kBinStride_ + kBinSizeY_));

  //   cv::imshow("GP variance heatmap", bad_region_heatmap_mask_);

  threshold(bad_region_heatmap_mask_,
            bad_region_heatmap_mask_,
            kNormalizedGPVarThresh,
            1.0,
            cv::THRESH_BINARY_INV);

  bad_region_heatmap_mask_.convertTo(bad_region_heatmap_mask_, CV_8U, 255.0);

  //   cv::imshow("GP heatmap mask", bad_region_heatmap_mask_);
  // --------------

  // Create a heatmap image of the bad regions for SLAM/VO
  cv::Mat bad_region_heatmap_low_res =
      GenerateErrHeatmap(bin_num_y, bin_num_x, grid_quality_vec, 1.0, 0.0);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  cv::resize(bad_region_heatmap_low_res,
             bad_region_heatmap_,
             cv::Size((bin_num_x - 1) * kBinStride_ + kBinSizeX_,
                      (bin_num_y - 1) * kBinStride_ + kBinSizeY_));

  bad_region_heatmap_.convertTo(bad_region_heatmap_, CV_8U, 255.0);
}

bool FeatureEvaluator::DrawReprojectionErrVec() {
  // Visualize only keypoints that have a large reprojection err
  const bool draw_only_bad_keypts = true;
  const int radius = 3;  // 5, 3

  Scalar green = Scalar(0, 255, 0);
  Scalar red = Scalar(0, 0, 255);
  Scalar blue = Scalar(255, 0, 0);

  if (error_type_ != kReprojection) {
    return false;
  }

  cvtColor(img_curr_, img_reproj_err_vec_, COLOR_GRAY2BGR);

  CHECK_EQ(err_vals_select_.size(), reproj2_select_.size());
  for (size_t i = 0; i < err_vals_select_.size(); i++) {
    if (err_vals_select_[i] <= kBadFeatureErrThresh_ && draw_only_bad_keypts) {
      continue;
    }

    Point point_st(static_cast<int>(reproj2_select_[i].x),
                   static_cast<int>(reproj2_select_[i].y));
    Point point_end(static_cast<int>(keypts2_select_[i].pt.x),
                    static_cast<int>(keypts2_select_[i].pt.y));

    circle(img_reproj_err_vec_, point_st, radius, red, -1, 8, 0);
    circle(img_reproj_err_vec_, point_end, radius, green, -1, 8, 0);
    arrowedLine(img_reproj_err_vec_, point_st, point_end, blue);
  }

  return true;
}

bool FeatureEvaluator::DrawEpipolarErrVec(bool was_recently_reset) {
  // Visualize only keypoints that have a large epipolar err
  const bool draw_only_bad_keypts = false;
  const int radius = 5;                     // 5, 3
  const float epipolar_line_seg_len = 800;  // pixels
  bool draw_epipolar_lines = false;
  bool draw_covariance = true;
  bool draw_sigma_pts_err = false;
  bool draw_keypts_life_ = false;
  bool draw_gt_reprojection = true;
  bool draw_bad_match_percent = true;
  bool draw_reset_notification = true;

  Scalar green = Scalar(0, 255, 0);
  Scalar red = Scalar(0, 0, 255);
  Scalar blue = Scalar(255, 0, 0);
  Scalar cyan = Scalar(255, 255, 0);
  Scalar white = Scalar(255, 255, 255);
  Scalar magenta = Scalar(255, 0, 255);
  Scalar brown = Scalar(0, 75, 150);

  if (error_type_ != kEpipolar || keypts2_select_.empty()) {
    return false;
  }

  // Draw covariances if they are available (the normalized epipolar error
  // calculation mode)
  if (!epi_err_vec_particles_cov_.empty()) {
    CHECK_EQ(keypts2_select_.size(), epi_err_vec_particles_cov_.size());
  }

  if (draw_sigma_pts_err && sigma_pts_err_.empty()) {
    draw_sigma_pts_err = false;
    LOG(WARNING) << "Sigma points error drawing requested but "
                 << "no sigma points are available!";
  }

  cvtColor(img_curr_, img_epipolar_err_vec_, COLOR_GRAY2BGR);

  CHECK_EQ(keypts2_select_.size(), epipolar_lines_dir_.size());
  CHECK_EQ(keypts2_select_.size(), epipolar_projections_.size());

  if (!reproj2_select_gt_.empty()) {
    CHECK_EQ(keypts2_select_.size(), reproj2_select_gt_.size());
  }

  for (size_t i = 0; i < keypts2_select_.size(); i++) {
    if (err_vals_select_[i] <= kBadFeatureErrThresh_ && draw_only_bad_keypts) {
      continue;
    }

    Point point_end(static_cast<int>(epipolar_projections_[i].x()),
                    static_cast<int>(epipolar_projections_[i].y()));

    Vector2f line_seg_st_eig = epipolar_projections_[i] -
                               epipolar_lines_dir_[i] * epipolar_line_seg_len;
    Point line_seg_st(static_cast<int>(line_seg_st_eig.x()),
                      static_cast<int>(line_seg_st_eig.y()));

    circle(img_epipolar_err_vec_, keypts2_select_[i].pt, radius, red, -1, 8, 0);
    circle(img_epipolar_err_vec_, point_end, radius, green, -1, 8, 0);
    arrowedLine(
        img_epipolar_err_vec_, keypts2_select_[i].pt, point_end, blue, 2, 8, 0);

    if (draw_epipolar_lines) {
      line(img_epipolar_err_vec_, line_seg_st, point_end, cyan);
    }

    if (draw_sigma_pts_err) {
      for (size_t j = 0; j < sigma_pts_err_[i].size(); j++) {
        cv::Point2f sigma_pt_err(sigma_pts_err_[i][j].x(),
                                 sigma_pts_err_[i][j].y());
        cv::Point2f sigma_pt(keypts2_select_[i].pt + sigma_pt_err);
        circle(img_epipolar_err_vec_, sigma_pt, radius, brown, -1, 8, 0);
        line(img_epipolar_err_vec_, keypts2_select_[i].pt, sigma_pt, brown);
      }

      // Calculate the variance of the absolute value of eppipolar error
      // for sigma points as opposed to the covariance of the epipolar error
      // vector
      //       vector<float> abs_err_values(sigma_pts_err_[i].size());
      float squared_sum = 0;
      for (size_t j = 0; j < sigma_pts_err_[i].size(); j++) {
        //        abs_err_values.push_back(sigma_pts_err_[i][j].norm());
        float abs_err = sigma_pts_err_[i][j].norm();
        squared_sum += abs_err * abs_err;
      }
      // The standard deviation
      float abs_err_std = sqrt(
          1.0 / static_cast<float>(sigma_pts_err_[i].size()) * squared_sum);
    }

    if (draw_covariance && !epi_err_vec_particles_cov_.empty()) {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(
          epi_err_vec_particles_cov_[i]);
      float l1 = solver.eigenvalues().x();
      float l2 = solver.eigenvalues().y();
      Vector2f e1 = solver.eigenvectors().col(0);
      Vector2f e2 = solver.eigenvectors().col(1);

      // 95% confidence interval
      double scale95 = sqrt(5.991);
      double R1 = scale95 * sqrt(l1);
      double R2 = scale95 * sqrt(l2);
      double tilt = atan2(e2.y(), e2.x()) * 180.0 / M_PI;

      cv::RotatedRect ellipse = cv::RotatedRect(point_end,
                                                cv::Size2f(2 * R1, 2 * R2),
                                                90.0 + tilt);  // -tilt

      cv::ellipse(img_epipolar_err_vec_, ellipse, magenta, 1);
    }

    if (draw_keypts_life_) {
      cv::Point2f cursor = keypts2_select_[i].pt;
      if (!keypts2_select_life_.empty() &&
          !keypts2_select_ref_frame_offset_.empty()) {
        int baseline = 0;
        double font_scale = 0.3;
        string keypt_life = to_string(keypts2_select_life_[i]);
        string keypt_ref_offset =
            to_string(keypts2_select_ref_frame_offset_[i]);
        //         string txt = keypt_life + "," + keypt_ref_offset;
        string txt = keypt_life;
        cv::Size txt_size =
            getTextSize(txt, FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);

        cursor.x -= txt_size.width / 2;
        cursor.y += txt_size.height + baseline;
        putText(img_epipolar_err_vec_,
                txt,
                cursor,
                FONT_HERSHEY_SIMPLEX,
                font_scale,
                blue,
                1);
      }
    }

    if (draw_gt_reprojection) {
      if (!reproj2_select_gt_.empty()) {
        CHECK_EQ(reproj2_select_gt_.size(), keypts2_select_.size());
        arrowedLine(img_epipolar_err_vec_,
                    keypts2_select_[i].pt,
                    reproj2_select_gt_[i],
                    blue,
                    2,
                    8,
                    0);
        cv::drawMarker(img_epipolar_err_vec_,
                       reproj2_select_gt_[i],
                       brown,
                       MARKER_CROSS,
                       10,
                       2,
                       8);
      }
    }
  }

  if (draw_bad_match_percent) {
    Scalar txt_color;
    if (IsFrameGoodForTraining()) {
      txt_color = green;
    } else {
      txt_color = red;
    }
    cv::Point2f cursor(480.0, 20.0);
    int baseline = 0;
    double font_scale = 0.8;
    string txt = "Bad Match: " + to_string(bad_matches_percent_) + "\%";
    cv::Size txt_size =
        getTextSize(txt, FONT_HERSHEY_SIMPLEX, font_scale, 3, &baseline);

    cursor.x -= txt_size.width / 2;
    cursor.y += txt_size.height + baseline;
    putText(img_epipolar_err_vec_,
            txt,
            cursor,
            FONT_HERSHEY_SIMPLEX,
            font_scale,
            txt_color,
            3);
  }

  if (draw_reset_notification && was_recently_reset) {
    string txt = "Restarted";
    double font_scale = 4.0;
    int thickness = 4;
    int baseline = 0;

    cv::Size txt_size = cv::getTextSize(
        txt, cv::FONT_HERSHEY_PLAIN, font_scale, thickness, &baseline);

    cv::putText(img_epipolar_err_vec_,
                txt,
                cv::Point(5, 20 + txt_size.height),
                cv::FONT_HERSHEY_PLAIN,
                font_scale,
                red,
                thickness,
                8);
  }

  return true;
}

void FeatureEvaluator::SaveImagesToFile(std::string target_path,
                                        const std::string& img_name,
                                        bool was_recently_reset) {
  const bool kDrawSelectedForTrainingFlag = true;
  const bool kSaveColoredHeatmaps = true;
  const bool kSaveColoredMaskedHeatmaps = true;
  static bool first_call = true;

  string img_name_truncated = img_name.substr(0, img_name.length() - 4);

  if (first_call) {
    RemoveDirectory(target_path + "/feature_qual/");
    RemoveDirectory(target_path + "/feature_matching/");
    RemoveDirectory(target_path + "/bad_matched_features/");
    RemoveDirectory(target_path + "/bad_region_heatmap/");
    RemoveDirectory(target_path + "/bad_region_heatmap_vis/");
    RemoveDirectory(target_path + "/bad_region_heatmap_masked_vis/");
    RemoveDirectory(target_path + "/reprojection_err_vec/");
    RemoveDirectory(target_path + "/epipolar_err_vec/");
    RemoveDirectory(target_path + "/err_norm_factor/");
    CreateDirectory(target_path);
    CreateDirectory(target_path + "/feature_qual/");
    CreateDirectory(target_path + "/feature_matching/");
    CreateDirectory(target_path + "/bad_matched_features/");
    CreateDirectory(target_path + "/bad_region_heatmap/");
    CreateDirectory(target_path + "/bad_region_heatmap_vis/");
    CreateDirectory(target_path + "/bad_region_heatmap_masked_vis/");
    CreateDirectory(target_path + "/reprojection_err_vec/");
    CreateDirectory(target_path + "/epipolar_err_vec/");
    CreateDirectory(target_path + "/err_norm_factor/");
  }

  string feature_qual_path =
      target_path + "/feature_qual/" + img_name_truncated + ".jpg";
  string feature_matching_path =
      target_path + "/feature_matching/" + img_name_truncated + ".jpg";
  string bad_matched_features_path =
      target_path + "/bad_matched_features/" + img_name_truncated + ".jpg";
  string bad_region_heatmap_path =
      target_path + "/bad_region_heatmap/" + img_name_truncated + ".jpg";
  string bad_region_heatmap_vis_path =
      target_path + "/bad_region_heatmap_vis/" + img_name_truncated + ".jpg";
  string bad_region_heatmap_masked_vis_path =
      target_path + "/bad_region_heatmap_masked_vis/" + img_name_truncated +
      ".jpg";
  string reproj_err_vec_path =
      target_path + "/reprojection_err_vec/" + img_name_truncated + ".jpg";
  string epipolar_err_vec_path =
      target_path + "/epipolar_err_vec/" + img_name_truncated + ".jpg";
  string err_norm_factor_path =
      target_path + "/err_norm_factor/" + img_name_truncated + ".jpg";

  cv::Mat tmp_img = GetFeatureErrVisualization();
  bool reproj_err_available = DrawReprojectionErrVec();
  bool epipolar_err_available = DrawEpipolarErrVec(was_recently_reset);

  // cv::imwrite(feature_qual_path, img_feature_qual_annotation_);
  // cv::imwrite(feature_matching_path, img_matching_annotation_);
  // cv::imwrite(bad_matched_features_path, img_bad_matching_annotation_);
  // cv::imwrite(bad_region_heatmap_path, bad_region_heatmap_);

  if (kSaveColoredHeatmaps || kSaveColoredMaskedHeatmaps) {
    Scalar flag_color;
    if (kDrawSelectedForTrainingFlag) {
      Scalar green = Scalar(0, 255, 0);
      Scalar red = Scalar(0, 0, 255);
      flag_color = (IsFrameGoodForTraining()) ? green : red;
    }

    cv::Mat heatmap_colored;
    cv::applyColorMap(bad_region_heatmap_, heatmap_colored, cv::COLORMAP_JET);

    cv::Mat heatmap_overlaid = OverlayHeatmapOnImage(heatmap_colored);

    if (kSaveColoredHeatmaps) {
      if (rectification_map_available_) {
        cv::Mat heatmap_overlaid_unrect = UnrectifyImage(heatmap_overlaid);
        if (kDrawSelectedForTrainingFlag) {
          circle(heatmap_overlaid_unrect,
                 Point(480, 20),
                 15,
                 flag_color,
                 -1,
                 8,
                 0);
        }
        cv::imwrite(bad_region_heatmap_vis_path, heatmap_overlaid_unrect);
      } else {
        if (kDrawSelectedForTrainingFlag) {
          circle(heatmap_overlaid, Point(480, 20), 15, flag_color, -1, 8, 0);
        }
        cv::imwrite(bad_region_heatmap_vis_path, heatmap_overlaid);
      }
    }

    if (kSaveColoredMaskedHeatmaps) {
      cv::Mat heatmap_overlaid_masked;
      heatmap_overlaid.copyTo(heatmap_overlaid_masked,
                              bad_region_heatmap_mask_);
      if (rectification_map_available_) {
        heatmap_overlaid_masked = UnrectifyImage(heatmap_overlaid_masked);
      }

      if (kDrawSelectedForTrainingFlag) {
        circle(
            heatmap_overlaid_masked, Point(480, 20), 15, flag_color, -1, 8, 0);
      }
      cv::imwrite(bad_region_heatmap_masked_vis_path, heatmap_overlaid_masked);
    }
  }

  if (reproj_err_available) {
    if (rectification_map_available_) {
      img_reproj_err_vec_ = UnrectifyImage(img_reproj_err_vec_);
    }
    cv::imwrite(reproj_err_vec_path, img_reproj_err_vec_);
  }

  if (epipolar_err_available) {
    if (rectification_map_available_) {
      img_epipolar_err_vec_ = UnrectifyImage(img_epipolar_err_vec_);
    }
    cv::imwrite(epipolar_err_vec_path, img_epipolar_err_vec_);
  }

  // Visualize the error normalization factors if available
  //   if (!err_norm_factor_.empty()) {
  //     ColorKeypoints(img_err_normalization_factor_,
  //                         keypts2_select_,
  //                         err_norm_factor_,
  //                         10000.0 * 0.005 * 0.064 , // 10000.0 * 0.005^2
  //                         5.0);
  //     cv::imwrite(err_norm_factor_path, img_err_normalization_factor_);
  //   }

  first_call = false;
}

bool FeatureEvaluator::GetGTReprojection(const ORB_SLAM2::Frame& ref_frame,
                                         const ORB_SLAM2::Frame& curr_frame,
                                         const int& keypt_idx_in_ref,
                                         const size_t& keypt_idx_in_curr,
                                         cv::Point2f* reprojection_pt) {
  if (ref_frame.mvKeysGTDepth.empty()) {
    return false;
  }
  cv::Point2f keypt(ref_frame.mvKeysUn[keypt_idx_in_ref].pt);
  float depth = ref_frame.mvKeysGTDepth[keypt_idx_in_ref];
  float x =
      depth * keypt.x / ref_frame.fx - depth * ref_frame.cx / ref_frame.fx;
  float y =
      depth * keypt.y / ref_frame.fy - depth * ref_frame.cy / ref_frame.fy;

  // Convert the point to current frame's coordinate frame

  // Convert the map point position to the homogeneous coordinate
  cv::Mat pt3d_ref = cv::Mat::ones(4, 1, CV_32F);
  pt3d_ref.at<float>(0) = x;
  pt3d_ref.at<float>(1) = y;
  pt3d_ref.at<float>(2) = depth;

  cv::Mat transform =
      CalculateRelativeTransform(curr_frame.mTwc_gt, ref_frame.mTwc_gt);

  cv::Mat pt3d_curr = transform * pt3d_ref;

  // If pt3d_curr is behind the camera, use the intersection of the line
  // segment that connects pt3d_curr to the matched point in current frame
  // with the plane z = epsilon to get the projection of pt3d_curr in current
  // frame. The result should be something that is outside the frame yet when
  // you draw a line segment on the image that connects that coordinate to the
  // matched point in current frame, it gives you an idea of in which direction
  // behind the camera is the ground truth location of the point
  cv::Mat pt3d_match_curr = cv::Mat::ones(4, 1, CV_32F);
  if (pt3d_curr.at<float>(2) < 0) {
    CHECK_EQ(curr_frame.mvKeysUn.size(), curr_frame.mvKeysGTDepth.size());

    cv::Point2f keypt(curr_frame.mvKeysUn[keypt_idx_in_curr].pt);
    float depth = curr_frame.mvKeysGTDepth[keypt_idx_in_curr];
    float x =
        depth * keypt.x / curr_frame.fx - depth * curr_frame.cx / curr_frame.fx;
    float y =
        depth * keypt.y / curr_frame.fy - depth * curr_frame.cy / curr_frame.fy;

    // Convert the map point position to the homogeneous coordinate
    pt3d_match_curr.at<float>(0) = x;
    pt3d_match_curr.at<float>(1) = y;
    pt3d_match_curr.at<float>(2) = depth;

    // The vector from pt3d_match_curr to pt3d_curr
    cv::Mat pt3d_match2orig = pt3d_curr - pt3d_match_curr;

    // Intersect pt3d_match_curr with the plane z = epsilon
    float epsilon = 1e-1;
    float d = (epsilon - depth) / (pt3d_match2orig.at<float>(2));

    cv::Mat intersection = pt3d_match_curr + d * pt3d_match2orig;
    pt3d_curr = intersection;
  }

  cv::Mat reproj_curr = ProjectToCam(curr_frame, pt3d_curr.rowRange(0, 3));

  reprojection_pt->x = reproj_curr.at<float>(0);
  reprojection_pt->y = reproj_curr.at<float>(1);

  return true;
}

bool FeatureEvaluator::GetGTReprojection(
    const ORB_SLAM2::KeyFrame& ref_keyframe,
    const ORB_SLAM2::Frame& curr_frame,
    const int& keypt_idx_in_ref,
    const size_t& keypt_idx_in_curr,
    cv::Point2f* reprojection_pt,
    bool* uncertain_gt_depth) {
  if (ref_keyframe.mvKeysGTDepth.empty()) {
    return false;
  }
  CHECK_EQ(ref_keyframe.mvKeysUn.size(), ref_keyframe.mvKeysGTDepth.size());

  cv::Point2f keypt(ref_keyframe.mvKeysUn[keypt_idx_in_ref].pt);
  float depth = ref_keyframe.mvKeysGTDepth[keypt_idx_in_ref];
  float x = depth * keypt.x / ref_keyframe.fx -
            depth * ref_keyframe.cx / ref_keyframe.fx;
  float y = depth * keypt.y / ref_keyframe.fy -
            depth * ref_keyframe.cy / ref_keyframe.fy;

  // Convert the point to current frame's coordinate frame

  // Convert the map point position to the homogeneous coordinate
  cv::Mat pt3d_ref = cv::Mat::ones(4, 1, CV_32F);
  pt3d_ref.at<float>(0) = x;
  pt3d_ref.at<float>(1) = y;
  pt3d_ref.at<float>(2) = depth;

  cv::Mat transform =
      CalculateRelativeTransform(curr_frame.mTwc_gt, ref_keyframe.mTwc_gt);

  cv::Mat pt3d_curr = transform * pt3d_ref;

  // If pt3d_curr is behind the camera, use the intersection of the line
  // segment that connects pt3d_curr to the matched point in current frame
  // with the plane z = epsilon to get the projection of pt3d_curr in current
  // frame. The result should be something that is outside the frame yet when
  // you draw a line segment on the image that connects that coordinate to the
  // matched point in current frame, it gives you an idea of in which direction
  // behind the camera is the ground truth location of the point
  cv::Mat pt3d_match_curr = cv::Mat::ones(4, 1, CV_32F);
  if (pt3d_curr.at<float>(2) < 0) {
    CHECK_EQ(curr_frame.mvKeysUn.size(), curr_frame.mvKeysGTDepth.size());

    cv::Point2f keypt(curr_frame.mvKeysUn[keypt_idx_in_curr].pt);
    float depth = curr_frame.mvKeysGTDepth[keypt_idx_in_curr];
    float x =
        depth * keypt.x / curr_frame.fx - depth * curr_frame.cx / curr_frame.fx;
    float y =
        depth * keypt.y / curr_frame.fy - depth * curr_frame.cy / curr_frame.fy;

    // Convert the map point position to the homogeneous coordinate
    pt3d_match_curr.at<float>(0) = x;
    pt3d_match_curr.at<float>(1) = y;
    pt3d_match_curr.at<float>(2) = depth;

    // The vector from pt3d_match_curr to pt3d_curr
    cv::Mat pt3d_match2orig = pt3d_curr - pt3d_match_curr;

    // Intersect pt3d_match_curr with the plane z = epsilon
    float epsilon = 1e-1;
    float d = (epsilon - depth) / (pt3d_match2orig.at<float>(2));

    cv::Mat intersection = pt3d_match_curr + d * pt3d_match2orig;
    pt3d_curr = intersection;
  }

  if (pt3d_curr.at<float>(2) > 600.0) {
    //     cout << "*******************************" << endl;
    //     LOG(WARNING) << "Point is too far!";
    //     cout << "*******************************" << endl;

    *uncertain_gt_depth = true;
  } else {
    *uncertain_gt_depth = false;
  }

  cv::Mat reproj_curr = ProjectToCam(curr_frame, pt3d_curr.rowRange(0, 3));

  reprojection_pt->x = reproj_curr.at<float>(0);
  reprojection_pt->y = reproj_curr.at<float>(1);

  return true;
}

// Uses Lie Algebra instead of euler angles
double FeatureEvaluator::CalculateNormalizedEpipolarErrorAnalytical(
    const cv::Mat& ref_frame_Twc_gt,
    const Eigen::Matrix<double, 6, 6>& ref_frame_cov_Twc_gt,
    const bool& ref_frame_cov_available,
    const std::string& ref_frame_name,
    const ORB_SLAM2::Frame& curr_frame,
    const cv::KeyPoint& keypoint1,
    const cv::KeyPoint& keypoint2,
    Eigen::Vector2f* epipolar_line_dir,
    Eigen::Vector2f* proj_on_epipolar_line,
    float* err_variance,
    double* err_norm_factor) {
  // If set to true, rejects evaluating keypoint pairs that are from frames
  // very close to each other
  const bool kAssertMinimumBaseLine = true;

  // TODO: remove this constant
  // The maximum accepted ratio for kAngualrVariance/||w|| and
  // kTranslationalVariance / t_prev_to_curr
  const float kMaxUncertaintyToRelativePoseRatio = 2.0;  // 2.0

  // The minimum accepted camera baseline
  const double kMinBaseLine = 0.03;  // meters

  float kAngualrVariance = 0.0;        // radians
  float kTranslationalVariance = 0.0;  // meters

  // If set to a positive value, the calculated error will be divided by this
  // constant. It is only to be used for the experimental case when
  // the camera pose estimates are noisy but no pose covariance is available.
  // (exp1unc)
  float kNormalizationFactor = 4.0;

  bool cam_pose_cov_available =
      ref_frame_cov_available && curr_frame.mbPoseUncertaintyAvailable;

  Eigen::Matrix<double, 6, 6> sigma_tf_prev_to_curr;

  // Covariance of the rotational part of relative transformation
  // tf_prev_to_curr
  Eigen::Matrix3d sigma_w;

  // Covariance of translation part of relative transformation tf_prev_to_curr
  Eigen::Matrix3d sigma_t;
  cv::Mat tf_prev_to_curr;

  if (cam_pose_cov_available) {
    tf_prev_to_curr = CalculateRelativeTransform(curr_frame.mTwc_gt,
                                                 ref_frame_Twc_gt,
                                                 curr_frame.mSigmaTwc_gt,
                                                 ref_frame_cov_Twc_gt,
                                                 &sigma_tf_prev_to_curr);
    kAngualrVariance = sigma_tf_prev_to_curr(0, 0);
    kTranslationalVariance = sigma_tf_prev_to_curr(5, 5);
    //     cout << "Estimated tf_prev_to_curr cov: " << endl;
    //     cout << sigma_tf_prev_to_curr << endl;

  } else if (rel_cam_pose_uncertainty_available_) {
    bool rel_pose_unc_retreived = GetRelativePoseUncertainty(
        ref_frame_name, curr_frame.mstrLeftImgName, &sigma_tf_prev_to_curr);
    if (!rel_pose_unc_retreived) {
      LOG(FATAL) << "Relative pose information was not found for the "
                 << "queried frames";
    }

    tf_prev_to_curr =
        CalculateRelativeTransform(curr_frame.mTwc_gt, ref_frame_Twc_gt);
  } else {
    tf_prev_to_curr =
        CalculateRelativeTransform(curr_frame.mTwc_gt, ref_frame_Twc_gt);
  }

  Vector3d trans_prev_to_curr(double(tf_prev_to_curr.at<float>(0, 3)),
                              double(tf_prev_to_curr.at<float>(1, 3)),
                              double(tf_prev_to_curr.at<float>(2, 3)));
  Matrix3d R_prev_to_curr;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      R_prev_to_curr(i, j) = double(tf_prev_to_curr.at<float>(i, j));
    }
  }

  if (kAssertMinimumBaseLine) {
    double trans_mag = trans_prev_to_curr.norm();
    if (trans_mag < kMinBaseLine) {
      LOG(INFO) << "Camera base line is too small. Skipping the data point.";
      return -1;
    }
  }

  // Calculates the unnormalized epipolar error
  float epipolar_err_scalar;
  Vector2f err_vec = CalculateEpipolarErrorVec(R_prev_to_curr,
                                               trans_prev_to_curr,
                                               curr_frame,
                                               keypoint1,
                                               keypoint2,
                                               &epipolar_err_scalar,
                                               epipolar_line_dir,
                                               proj_on_epipolar_line);

  if (cam_pose_cov_available || rel_cam_pose_uncertainty_available_) {
    sigma_w = sigma_tf_prev_to_curr.topLeftCorner(3, 3);
    sigma_t = sigma_tf_prev_to_curr.bottomRightCorner(3, 3);
  } else {
    sigma_w << kAngualrVariance, 0, 0, 0, kAngualrVariance, 0, 0, 0,
        kAngualrVariance;

    sigma_t << kTranslationalVariance, 0, 0, 0, kTranslationalVariance, 0, 0, 0,
        kTranslationalVariance;
  }

  // Get the Jacobians of epipolar error w.r.t. perturbations in the
  // transformation between the two frames
  Eigen::RowVector3d J_t, J_w;
  Eigen::Vector3d x_ref(static_cast<double>(keypoint1.pt.x),
                        static_cast<double>(keypoint1.pt.y),
                        1.0);
  Eigen::Vector3d x(static_cast<double>(keypoint2.pt.x),
                    static_cast<double>(keypoint2.pt.y),
                    1.0);
  GetEpipolarErrorJacobians(
      curr_frame, R_prev_to_curr, trans_prev_to_curr, x_ref, x, &J_w, &J_t);

  double var_w = J_w * sigma_w * J_w.transpose();
  double var_t = J_t * sigma_t * J_t.transpose();
  double var = var_w + var_t;

  //   cout << "var_w, var_t: " << var_w << ", " << var_t << endl;

  // The uncertainty in the location of keypoint given its size
  float keypt_sigma2 = 1 / curr_frame.mvInvLevelSigma2[keypoint2.octave];

  //   cout << "var: " << var << ", keypt_sigma: " << keypt_sigma2 << endl;

  // Combine the uncertainty due to the keypoint size and the uncertainty
  // due to inaccurate camera pose estimate
  var = var + keypt_sigma2;

  *err_variance = var;

  // 95% confidence interval
  float scale95 = sqrt(5.991);
  float normalization_factor = scale95 * static_cast<float>(sqrt(var));

  if (kNormalizationFactor > 0 && !cam_pose_cov_available &&
      !rel_cam_pose_uncertainty_available_) {
    normalization_factor *= kNormalizationFactor;
  }

  if (var < 0) {
    LOG(INFO) << "Invalid covariance matrix determinant value:" << var << endl;
  }

  if (kFilterBasedOnEpipolarErrInformation_) {
    if (normalization_factor < kMinEpipolarErrSensitivity_) {
      LOG(INFO) << "Removed insensitive point!";
      return -1;
    }
  }

  // Enforce a minimum normalization scalar to prevent numerical errors.
  normalization_factor = (normalization_factor < kMinEpipolarErrSensitivity_)
                             ? kMinEpipolarErrSensitivity_
                             : normalization_factor;
  //   normalization_factor *= (keypoint2.size);
  *err_norm_factor = static_cast<double>(normalization_factor);

  return epipolar_err_scalar / static_cast<double>(normalization_factor);
  //   return epipolar_err_scalar;
}

// Uses Lie Algebra instead of euler angles
double FeatureEvaluator::CalculateNormalizedEpipolarError(
    const cv::Mat& ref_frame_Twc_gt,
    const ORB_SLAM2::Frame& curr_frame,
    const cv::KeyPoint& keypoint1,
    const cv::KeyPoint& keypoint2,
    Eigen::Vector2f* epipolar_line_dir,
    Eigen::Vector2f* proj_on_epipolar_line,
    std::vector<Eigen::Vector2f>* sigma_pts_err,
    Eigen::Matrix2f* err_covariance,
    double* err_norm_factor) {
  const float kAngualrVariancePct = 0.001;        // percentage
  const float kTranslationalVariancePct = 0.001;  // percentage

  const float kAngualrVariance = 0.0;        // radians : 2deg
  const float kTranslationalVariance = 0.0;  // meters : 0.1m

  // When set to true the epipolar error of the sigma points are calculated
  // with respect to the projection of keypoint2 on the original epipolar line
  // This is to prevent a larger normalization factor in cases when there
  // exists a feature matching error.
  const bool kCalcEpipolarCovWRTKeyPt2Projection = true;

  sigma_pts_err->clear();

  // The calculated covariance of the epipolar error wrt the ground truth
  // transformation uncertainty is normalized with this coefficient.
  const float kCovarianceNormalizationCoeff = 1.0;
  //   const float kCovarianceNormalizationCoeff = projection_mat_cam0_(0,0);

  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, ref_frame_Twc_gt);
  Vector3d trans_prev_to_curr(double(tf_prev_to_curr.at<float>(0, 3)),
                              double(tf_prev_to_curr.at<float>(1, 3)),
                              double(tf_prev_to_curr.at<float>(2, 3)));
  Matrix3d R_prev_to_curr;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      R_prev_to_curr(i, j) = double(tf_prev_to_curr.at<float>(i, j));
    }
  }

  // Calculates the unnormalized epipolar error
  float epipolar_err_scalar;
  Vector2f err_vec = CalculateEpipolarErrorVec(R_prev_to_curr,
                                               trans_prev_to_curr,
                                               curr_frame,
                                               keypoint1,
                                               keypoint2,
                                               &epipolar_err_scalar,
                                               epipolar_line_dir,
                                               proj_on_epipolar_line);

  // Generate a set of sigma points (for the relative pose of the camera) and
  // calculate the epipolar error vector for each of them

  // The state vector is [rotx, roty, rotz, tx, ty, tz]T

  // Convert the rotation matrix to euler angles
  Vector3d euler_angles = R_prev_to_curr.eulerAngles(0, 1, 2);

  // The mean transformation: [rotx, roty, rotz, tx, ty, tz]T
  //   Vector6f mean(euler_angles(0).cast<float>(),
  //                                   euler_angles(1).cast<float>(),
  //                                   euler_angles(2).cast<float>(),
  //                                   trans_prev_to_curr.cast<float>());

  Eigen::Matrix<float, 6, 1> mean;
  mean << euler_angles.cast<float>(), trans_prev_to_curr.cast<float>();

  // Covariance matrix:
  Eigen::Matrix<float, 6, 6> covariance;
  covariance << kAngualrVariance, 0, 0, 0, 0, 0, 0, kAngualrVariance, 0, 0, 0,
      0, 0, 0, kAngualrVariance, 0, 0, 0, 0, 0, 0, kTranslationalVariance, 0, 0,
      0, 0, 0, 0, kTranslationalVariance, 0, 0, 0, 0, 0, 0,
      kTranslationalVariance;

  // Covariance matrix using the percentage error
  //   covariance << kAngualrVariancePct * fabs(mean(0)),       0, 0, 0, 0, 0,
  //                 0, kAngualrVariancePct * fabs(mean(1)) ,      0, 0, 0, 0,
  //                 0, 0, kAngualrVariancePct * fabs(mean(2)),       0, 0, 0,
  //                 0, 0, 0, kTranslationalVariancePct * fabs(mean(3)), 0, 0,
  //                 0, 0, 0, 0, kTranslationalVariancePct * fabs(mean(4)), 0,
  //                 0, 0, 0, 0, 0, kTranslationalVariancePct * fabs(mean(5));

  // Dimension of the state vector
  int n = mean.rows();
  // TODO: Tune lambda coeff
  float lambda_coeff = static_cast<float>(3 - n);

  // For each sigma point generate the corresponding particle that consists of
  // rotation matrix and translation vector
  // 2n + 1 Sigma points are generated given the formula:
  // sigma_pts[0] = mean
  // sigma_pts[i] = mean + sqrt((n + lambda_coeff) * covariance(:, i - 1))
  //                                                      for i = 1, ..., n
  // sigma_pts[i] = mean - sqrt((n + lambda_coeff) * covariance(:, i - n - 1))
  //                                                      for i = n + 1, ..., 2n
  // weights[0] = lambda_coeff/ (n + lambda_coeff)
  // weights[i] = 1 / 2(n + lambda_coeff) for i = 1, ..., 2n

  // Vector of sigma point weights
  vector<float> weights(2 * n + 1);
  vector<Eigen::Matrix<float, 6, 1>> sigma_pts;

  // Vector of epipolar error vectors for all particles
  vector<Vector2f> particles_err_vec(2 * n + 1);

  //   weights[0] = lambda_coeff / (n + lambda_coeff);
  weights[0] = 1.0 / (2.0 * n + 1.0);

  sigma_pts.push_back(mean);

  if (kCalcEpipolarCovWRTKeyPt2Projection) {
    particles_err_vec[0] = Vector2f(0, 0);
  } else {
    particles_err_vec[0] = err_vec;
  }

  for (size_t i = 0; i < static_cast<size_t>(2 * n); i++) {
    Eigen::Matrix<float, 6, 1> sigma_pt;

    if (i < n) {
      //       sigma_pt = mean.array()
      //                   + ((n + lambda_coeff) *
      //                   covariance.col(i)).array().sqrt();
      sigma_pt = mean.array() + (covariance.col(i)).array().sqrt();
    } else {
      //       sigma_pt = mean.array() -
      //                  ((n + lambda_coeff) * covariance.col(i -
      //                  n)).array().sqrt();
      sigma_pt = mean.array() - (covariance.col(i - n)).array().sqrt();
    }

    //     weights[i + 1] = 1.0 / (2 * (n + lambda_coeff));
    weights[i + 1] = 1.0 / (2.0 * n + 1.0);

    sigma_pts.push_back(sigma_pt);

    Eigen::Matrix3f R_prev_to_curr_particle;
    R_prev_to_curr_particle = AngleAxisf(sigma_pt(0), Vector3f::UnitX()) *
                              AngleAxisf(sigma_pt(1), Vector3f::UnitY()) *
                              AngleAxisf(sigma_pt(2), Vector3f::UnitZ());
    Vector3f trans_prev_to_curr_particle(sigma_pt(3), sigma_pt(4), sigma_pt(5));

    cv::KeyPoint keypoint2_proxy(keypoint2);
    if (kCalcEpipolarCovWRTKeyPt2Projection) {
      keypoint2_proxy.pt.x = proj_on_epipolar_line->x();
      keypoint2_proxy.pt.y = proj_on_epipolar_line->y();
    }

    float err_scalar_particle;
    Vector2f epipolar_line_dir_particle;
    Vector2f proj_on_epipolar_line_particle;
    Vector2f err_vec_particle =
        CalculateEpipolarErrorVec(R_prev_to_curr_particle.cast<double>(),
                                  trans_prev_to_curr_particle.cast<double>(),
                                  curr_frame,
                                  keypoint1,
                                  keypoint2_proxy,
                                  &err_scalar_particle,
                                  &epipolar_line_dir_particle,
                                  &proj_on_epipolar_line_particle);
    particles_err_vec[i + 1] = err_vec_particle;
  }

  // Debugging: Overwrite particles err_vec for debugging the calculated
  // cov elliplse:
  //   particles_err_vec[0] = Vector2f(0, 0);
  //   particles_err_vec[1] = Vector2f(40, 0);
  //   particles_err_vec[2] = Vector2f(-40, 0);
  //   particles_err_vec[3] = Vector2f(0, 5);
  //   particles_err_vec[4] = Vector2f(0, -5);
  //   particles_err_vec[5] = Vector2f(40, 0);
  //   particles_err_vec[6] = Vector2f(-40, 0);
  //   particles_err_vec[7] = Vector2f(0, 5);
  //   particles_err_vec[8] = Vector2f(0, -5);
  //   particles_err_vec[9] = Vector2f(40, 0);
  //   particles_err_vec[10] = Vector2f(-40, 0);
  //   particles_err_vec[11] = Vector2f(0, 5);
  //   particles_err_vec[12] = Vector2f(0, -5);
  //   float rotation_angle = M_PI / 4.0;
  //   for (size_t i = 0; i < static_cast<size_t>(2 * n); i++) {
  //     float x_rot = cos(rotation_angle) * particles_err_vec[i].x()
  //                         -sin(rotation_angle) * particles_err_vec[i].y();
  //     float y_rot = sin(rotation_angle) * particles_err_vec[i].x()
  //                         +cos(rotation_angle) * particles_err_vec[i].y();
  //
  //     particles_err_vec[i].x() = x_rot;
  //     particles_err_vec[i].y() = y_rot;
  //
  //   }

  *sigma_pts_err = particles_err_vec;

  // Calculate the predicted mean and covariance of the epipolar error vector
  Vector2f predicted_mean(0, 0);
  for (size_t i = 0; i < sigma_pts.size(); i++) {
    predicted_mean += weights[i] * particles_err_vec[i];
  }

  Eigen::Matrix2f predicted_covariance = Eigen::Matrix2f::Zero();
  Eigen::Matrix<long double, 2, 2> pred_cov_long;
  for (size_t i = 0; i < sigma_pts.size(); i++) {
    predicted_covariance += weights[i] *
                            (particles_err_vec[i] - predicted_mean) *
                            (particles_err_vec[i] - predicted_mean).transpose();
  }

  // Calculate the variance of the absolute value of eppipolar error
  // for sigma points as opposed to the covariance of the epipolar error
  // vector (since the absolute value is what is used as quality metric)
  float squared_sum = 0;
  for (size_t j = 0; j < particles_err_vec.size(); j++) {
    float abs_err = particles_err_vec[j].norm();
    squared_sum += abs_err * abs_err;
  }
  // The standard deviation
  float abs_err_std =
      sqrt(1.0 / static_cast<float>(particles_err_vec.size()) * squared_sum);

  // +++++
  // The standard deviation of the absolute value of epipolar error
  // of sigma points AFTER DEDUCTING THE MEAN ERR vector
  float squared_sum_m = 0;
  for (size_t j = 0; j < particles_err_vec.size(); j++) {
    float abs_err = (particles_err_vec[j] - predicted_mean).norm();
    squared_sum_m += abs_err * abs_err;
  }
  float abs_err_std_m =
      sqrt(1.0 / static_cast<float>(particles_err_vec.size()) * squared_sum_m);

  // +++++

  // Debugging printouts ********
  if (false) {
    cout << "covariance: " << endl;
    cout << covariance << endl;
    cout << "Original error vector: " << err_vec << endl;
    cout << "Predicted mean error vector: " << predicted_mean << endl;
    cout << "Predicted covariance: " << predicted_covariance << endl;

    //   cout << "long cov det: " << pred_cov_long.determinant() << endl;
    //   cout << "float cov det: " << predicted_covariance.determinant() <<
    //   endl;

    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    cout << "Sigma Points: " << endl;
    for (size_t i = 0; i < sigma_pts.size(); i++) {
      cout << sigma_pts[i].transpose() << endl;
    }
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;

    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    cout << "Sigma Pts Errors: " << endl;
    for (size_t i = 0; i < sigma_pts.size(); i++) {
      cout << (particles_err_vec[i] - predicted_mean).norm() << endl;
    }
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
  }

  // ******************* ********

  // Scales the epipolar error magnitude given the error covariance
  // The nth root of determinent is used here but here are a list of other
  // measures that could be used:
  // https://stats.stackexchange.com/questions/225434/
  // a-measure-of-variance-from-the- covariance-matrix

  *err_covariance = predicted_covariance;

  // TODO: Add back the division by the variance measure, but make sure
  // the resultant normalized error range is reasonable.
  if (predicted_covariance.determinant() < 0) {
    LOG(INFO) << "Invalid covariance matrix determinant value:"
              << predicted_covariance.determinant() << endl;
  }

  // Use nth root of the covariance matrix determinant (multiplication
  // of all eigen values) as the normnalization factor
  //   float normalization_factor = sqrt(predicted_covariance.determinant())
  //                                 / kCovarianceNormalizationCoeff;

  // Use the largest eigen value of the covariance matrix as the normalization
  // factor
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(predicted_covariance);
  float l1 = solver.eigenvalues().x();
  float l2 = solver.eigenvalues().y();
  float normalization_factor = max(l1, l2) / kCovarianceNormalizationCoeff;

  // Use the standard deviation of the absolute epipolar error value of the
  // sigma points as the normalization factor
  //   float normalization_factor = abs_err_std;

  // Use the the standard deviation of the absolute value of epipolar error
  // of sigma points AFTER DEDUCTING THE MEAN ERR vector
  //   float normalization_factor = abs_err_std_m;

  //   cout << "fx, fy: " << projection_mat_cam0_(0,0) << ", "
  //        << projection_mat_cam0_(1,1) << endl;
  //   cout << "Normalized Error Value: "
  //        << epipolar_err_scalar / normalization_factor
  //        << endl;
  //   cout << "covariance det: " << sqrt(predicted_covariance.determinant())
  //        << endl;
  //   cout << "normalized covariance det: " <<
  //           normalization_factor << endl;
  //   cout << "epipolar err: " << epipolar_err_scalar << endl;

  if (kFilterBasedOnEpipolarErrInformation_) {
    if (normalization_factor < kMinEpipolarErrSensitivity_) {
      LOG(INFO) << "Removed insensitive point!";
      return -1;
    }
  }

  // Enforce a minimum normalization scalar to prevent numerical errors.
  normalization_factor = (normalization_factor < kMinEpipolarErrSensitivity_)
                             ? kMinEpipolarErrSensitivity_
                             : normalization_factor;
  normalization_factor *= (keypoint2.size);
  *err_norm_factor = static_cast<double>(normalization_factor);

  //   return epipolar_err_scalar;
  return epipolar_err_scalar / static_cast<double>(normalization_factor);
}

void FeatureEvaluator::GetEpipolarErrorJacobians(
    const ORB_SLAM2::Frame& cam_frame,
    const Eigen::Matrix3d& R_prev_to_curr,
    const Eigen::Vector3d& trans_prev_to_curr,
    const Eigen::Vector3d& x_ref,
    const Eigen::Vector3d& x,
    Eigen::RowVector3d* J_w,
    Eigen::RowVector3d* J_t) {
  // Normalize x_ref
  Eigen::Vector3d x_ref_n = x_ref / x_ref(2);
  Eigen::Vector3d x_n = x / x(2);

  Eigen::Matrix3d cam_mat;
  cam_mat << cam_frame.fx, 0.0, cam_frame.cx, 0.0, cam_frame.fy, cam_frame.cy,
      0.0, 0.0, 1.0;

  // The jacobians of the epipolar line
  Eigen::Matrix3d Jl_w, Jl_t;
  GetEpipolarLineJacobians(
      cam_mat, R_prev_to_curr, trans_prev_to_curr, x_ref_n, &Jl_w, &Jl_t);

  // Calculate the fundamental matrix
  Eigen::Matrix3d F =
      GetFundamentalMatrix(cam_mat, R_prev_to_curr, trans_prev_to_curr);

  // Epipolar line on current frame
  Eigen::Vector3d l = F * x_ref_n;
  double L = l.topLeftCorner(2, 1).norm();
  double L_cube = L * L * L;

  *J_t = (x_n.transpose() * Jl_t / L) -
         (x_n.transpose() * l) * (l(0) * Jl_t.row(0) + l(1) * Jl_t.row(1)) /
             L_cube;

  *J_w = (x_n.transpose() * Jl_w / L) -
         (x_n.transpose() * l) * (l(0) * Jl_w.row(0) + l(1) * Jl_w.row(1)) /
             L_cube;
}

void FeatureEvaluator::GetEpipolarLineJacobians(
    const Eigen::Matrix3d& cam_mat,
    const Eigen::Matrix3d& R_prev_to_curr,
    const Eigen::Vector3d& trans_prev_to_curr,
    const Eigen::Vector3d& x_ref,
    Eigen::Matrix3d* J_w,
    Eigen::Matrix3d* J_t) {
  // Normalize x_ref
  Eigen::Vector3d x_ref_n = x_ref / x_ref(2);

  // B = K *  R * inv(K) * x_r;
  Eigen::Vector3d B = cam_mat * R_prev_to_curr * cam_mat.inverse() * x_ref_n;

  for (int i = 0; i < 3; i++) {
    J_t->col(i) = cam_mat.col(i).cross(B);
  }

  Eigen::Vector3d C = R_prev_to_curr * cam_mat.inverse() * x_ref_n;
  Eigen::Matrix3d C_x = GetSkewSymmetric(C);
  Eigen::Matrix3d E = GetSkewSymmetric(cam_mat * trans_prev_to_curr);

  *J_w = -E * cam_mat * C_x;
}

Eigen::Matrix3d FeatureEvaluator::GetFundamentalMatrix(
    const Eigen::Matrix3d& cam_mat,
    const Eigen::Matrix3d& R_prev_to_curr,
    const Eigen::Vector3d& trans_prev_to_curr) {
  Eigen::Vector3d F_A = cam_mat * trans_prev_to_curr;
  Eigen::Matrix3d F_B = cam_mat * R_prev_to_curr * cam_mat.inverse();

  return GetSkewSymmetric(F_A) * F_B;
}

Eigen::Matrix3d FeatureEvaluator::GetSkewSymmetric(Eigen::Vector3d vec) {
  Eigen::Matrix3d skew_symm;
  skew_symm << 0.0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0;
  return skew_symm;
}

inline float Kernel(const Vector2f& x1, const Vector2f& x2) {
  //   const float s_f = 2.8636;
  //   const float l = 17.8;
  const float s_f = 80.0;
  const float l = 100.0;  // def: 200

  return s_f * s_f * exp(-1.0 / (2.0 * l * l) * ((x1 - x2).squaredNorm()));
}

inline float Kernel(const Vector2f& x1,
                    const Vector2f& x2,
                    float s_f,
                    float l) {
  return s_f * s_f * exp(-1.0 / (2.0 * l * l) * ((x1 - x2).squaredNorm()));
}

MatrixXf FeatureEvaluator::Kmatrix(const vector<Vector2f>& X) {
  //   const float s_n = 2.0;
  const float s_n = 20.0;

  int N = X.size();
  MatrixXf Km(N, N), I(N, N);
  I.setIdentity();
  I.setIdentity();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      Km(i, j) = Kernel(X[i], X[j]);
      Km(j, i) = Km(i, j);
    }
  }
  Km = (Km + s_n * s_n * I).inverse();
  return Km;
}

void FeatureEvaluator::GPPredict(float x,
                                 float y,
                                 const vector<Vector2f>& locs,
                                 const VectorXf& values,
                                 const MatrixXf& K_mat,
                                 float& mean,
                                 float& variance) {
  const float kErrMean = kErrMinClamp_;

  int N = locs.size();
  MatrixXf Kv(N, 1), Kvt;
  Vector2f l(x, y);
  for (int i = 0; i < N; i++) {
    Kv(i, 0) = Kernel(l, locs[i]);
  }
  Kvt = Kv.transpose();
  MatrixXf u = Kvt * K_mat;
  MatrixXf ret;
  ret = u * values;
  mean = ret(0, 0) + kErrMean;
  ret = (u * Kv);
  variance = ret(0, 0);
  variance = Kernel(l, l) - variance;
}

bool FeatureEvaluator::IsFrameGoodForTraining() {
  if (frame_reliability_ == Unknown) {
    if (bad_matches_percent_ < kMaxBadMatchPercent_ &&
        bad_matches_percent_ > kMinBadMatchPercent_ &&
        err_vals_select_.size() > kMinMatchesInFrame_) {
      return true;
    } else {
      return false;
    }
  } else {
    if (frame_reliability_ == Reliable) {
      return true;
    } else {
      return false;
    }
  }
}

void FeatureEvaluator::SetFrameReliability(Reliability frame_reliability) {
  frame_reliability_ = frame_reliability;
}

void FeatureEvaluator::SetRelativeCamPoseUncertainty(
    const std::unordered_map<std::string, int>* pose_unc_map,
    const std::vector<Eigen::Vector2f>* rel_cam_poses_uncertainty) {
  rel_cam_pose_uncertainty_available_ = true;
  rel_cam_poses_unc_map_ = pose_unc_map;
  rel_cam_poses_unc_ = rel_cam_poses_uncertainty;
}

bool FeatureEvaluator::GetRelativePoseUncertainty(
    const std::string& ref_frame_name,
    const std::string& curr_frame_name,
    Eigen::Matrix<double, 6, 6>* rel_pos_cov) {
  if (!rel_cam_pose_uncertainty_available_) {
    cout << "rel_cam_pose_uncertainty not available!" << endl;
    return false;
  }

  std::unordered_map<std::string, int>::const_iterator it_ref =
      rel_cam_poses_unc_map_->find(ref_frame_name);
  std::unordered_map<std::string, int>::const_iterator it_curr =
      rel_cam_poses_unc_map_->find(curr_frame_name);

  //   it_curr = rel_cam_poses_unc_map_->find(curr_frame_name);
  if (it_ref == rel_cam_poses_unc_map_->end() ||
      it_curr == rel_cam_poses_unc_map_->end()) {
    return false;
  }

  int ref_frame_id = it_ref->second;
  int curr_frame_id = it_curr->second;

  double scale95 = sqrt(5.991);

  // Naive Approach: Take the maximum translational and rotational uncertainty
  // independently and form the covariance matrix
  float max_trans_unc = std::numeric_limits<float>::min();
  float max_rot_unc = std::numeric_limits<float>::min();

  for (int i = ref_frame_id; i <= curr_frame_id; i++) {
    if (rel_cam_poses_unc_->at(i)(0) > max_trans_unc) {
      max_trans_unc = rel_cam_poses_unc_->at(i)(0);
    }

    if (rel_cam_poses_unc_->at(i)(1) > max_rot_unc) {
      max_rot_unc = rel_cam_poses_unc_->at(i)(1);
    }
  }

  float tran_unc_scalar = 4.0;
  float rot_unc_scalar = 0.1;
  max_trans_unc /= tran_unc_scalar;
  max_rot_unc /= rot_unc_scalar;

  rel_pos_cov->setIdentity();
  rel_pos_cov->bottomRightCorner(3, 3) *=
      (max_trans_unc / scale95) * (max_trans_unc / scale95);
  rel_pos_cov->topLeftCorner(3, 3) *=
      (max_rot_unc / (scale95 * sqrt(3))) * (max_rot_unc / (scale95 * sqrt(3)));

  //   cout << *rel_pos_cov << endl << endl;

  return true;
}

void FeatureEvaluator::CalcRelativePoseError(cv::Mat& pose_est_0,
                                             cv::Mat& pose_est_1,
                                             cv::Mat& pose_gt_0,
                                             cv::Mat& pose_gt_1,
                                             Eigen::AngleAxisd* aa_rot_err,
                                             Eigen::Vector3d* t_err) {
  // Map to Eigen Matrices
  Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_est0(
      pose_est_0.ptr<float>(), pose_est_0.rows, pose_est_0.cols);
  Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_est1(
      pose_est_1.ptr<float>(), pose_est_1.rows, pose_est_1.cols);
  Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_gt0(
      pose_gt_0.ptr<float>(), pose_gt_0.rows, pose_gt_0.cols);
  Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_gt1(
      pose_gt_1.ptr<float>(), pose_gt_1.rows, pose_gt_1.cols);

  // The estimated pose of the camera at the end of prediction horizon
  Eigen::Vector3f t_est_H = T_est1.topRightCorner(3, 1);
  Eigen::Matrix3f R_est_H = T_est1.topLeftCorner(3, 3);

  Eigen::Vector3f t_est_curr = T_est0.topRightCorner(3, 1);
  Eigen::Matrix3f R_est_curr = T_est0.topLeftCorner(3, 3);

  Eigen::Vector3f t_H = T_gt1.topRightCorner(3, 1);
  Eigen::Matrix3f R_H = T_gt1.topLeftCorner(3, 3);

  Eigen::Vector3f t_curr = T_gt0.topRightCorner(3, 1);
  Eigen::Matrix3f R_curr = T_gt0.topLeftCorner(3, 3);

  // The reference tranformation from frame H to current frame
  Eigen::Vector3f t_ref_curr_H;
  Eigen::Matrix3f R_ref_curr_H;

  R_ref_curr_H = R_curr.transpose() * R_H;
  t_ref_curr_H = R_curr.transpose() * t_H - R_curr.transpose() * t_curr;

  Eigen::Vector3f t_inv_ref_curr_H = -R_ref_curr_H.transpose() * t_ref_curr_H;
  Eigen::Matrix3f R_inv_ref_curr_H = R_ref_curr_H.transpose();

  // The estimated tranformation from frame H to current frame
  Eigen::Vector3f t_est_curr_H;
  Eigen::Matrix3f R_est_curr_H;

  R_est_curr_H = R_est_curr.transpose() * R_est_H;
  t_est_curr_H =
      R_est_curr.transpose() * t_est_H - R_est_curr.transpose() * t_est_curr;

  *aa_rot_err = (R_inv_ref_curr_H * R_est_curr_H).cast<double>();
  *t_err = (t_inv_ref_curr_H + R_inv_ref_curr_H * t_est_curr_H).cast<double>();
}

void FeatureEvaluator::EvaluateAgainstPrevFrame(ORB_SLAM2::Frame& prev_frame,
                                                ORB_SLAM2::Frame& curr_frame) {
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, prev_frame.mTwc_gt);
  float scale = 1.0;
  tf_prev_to_curr.rowRange(0, 3).col(3) =
      tf_prev_to_curr.rowRange(0, 3).col(3) / scale;

  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = curr_frame.mvpMapPoints[i];
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;

      if (prev_frame.GetCorrespondingKeyPt(curr_map_pt, &keypt_prev)) {
        keypts2_select_.push_back(keypt_curr);
        keypts2_matched_w_prev_.push_back(keypt_curr);
        keypts1_matched_w_curr_.push_back(keypt_prev);
        int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
        matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
        cv::Point2f reproj_pt;
        float reproj_err = CalculateReprojectionError(prev_frame,
                                                      curr_frame,
                                                      tf_prev_to_curr,
                                                      *curr_map_pt,
                                                      keypt_curr,
                                                      &reproj_pt);
        err_vals_select_.push_back(reproj_err);
        reproj2_select_.push_back(reproj_pt);
        err_vals_visualization_.push_back(reproj_err);
      }
    }
  }
}

void FeatureEvaluator::EvaluateAgainstPrevFrameAndLastKeyFrame(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, prev_frame.mTwc_gt);
  float scale = 1.0;
  tf_prev_to_curr.rowRange(0, 3).col(3) =
      tf_prev_to_curr.rowRange(0, 3).col(3) / scale;

  LOG(FATAL) << "This training mode is not yet implemented!";
}

void FeatureEvaluator::EvaluateAgainstPrevFrameAndRefKeyFrame(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, prev_frame.mTwc_gt);
  float scale = 1.0;
  tf_prev_to_curr.rowRange(0, 3).col(3) =
      tf_prev_to_curr.rowRange(0, 3).col(3) / scale;

  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = curr_frame.mvpMapPoints[i];
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;

      keypts2_select_.push_back(keypt_curr);
      if (prev_frame.GetCorrespondingKeyPt(curr_map_pt, &keypt_prev)) {
        keypts2_matched_w_prev_.push_back(keypt_curr);
        keypts1_matched_w_curr_.push_back(keypt_prev);
        int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
        matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
        cv::Point2f reproj_pt;
        float reproj_err = CalculateReprojectionError(prev_frame,
                                                      curr_frame,
                                                      tf_prev_to_curr,
                                                      *curr_map_pt,
                                                      keypt_curr,
                                                      &reproj_pt);
        err_vals_select_.push_back(reproj_err);
        reproj2_select_.push_back(reproj_pt);
        err_vals_visualization_.push_back(reproj_err);
      } else {
        // Reference frame of current map point
        KeyFrame* pt_ref_keyframe = curr_map_pt->GetReferenceKeyFrame();
        cv::Point2f reproj_pt;
        float reproj_err = CalculateReprojectionError(
            *pt_ref_keyframe, curr_frame, *curr_map_pt, keypt_curr, &reproj_pt);
        err_vals_select_.push_back(reproj_err);
        reproj2_select_.push_back(reproj_pt);
      }
    }
  }
}

void FeatureEvaluator::EvaluateAgainstLastKeyFrame(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  LOG(FATAL) << "This training mode is not implemented!";
}

void FeatureEvaluator::EvaluateAgainstRefKeyFrame(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = curr_frame.mvpMapPoints[i];
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;
      keypts2_select_.push_back(keypt_curr);

      // Reference frame of current map point
      KeyFrame* pt_ref_keyframe = curr_map_pt->GetReferenceKeyFrame();
      cv::Point2f reproj_pt;
      float reproj_err = CalculateReprojectionError(
          *pt_ref_keyframe, curr_frame, *curr_map_pt, keypt_curr, &reproj_pt);
      err_vals_select_.push_back(reproj_err);
      reproj2_select_.push_back(reproj_pt);

      // Since for now only the current frame and the previous frame
      // images are available, for visualizing matched keypoints, only
      // show those that are seen in the previous frame as well as
      // current frame.
      if (false) {
        // Find the index of the corresponding descriptor in the reference
        // keyframe of current frame. It is only for visualization purposes.
        // Here keypts1_matched_w_curr_ is filled with keypoints in the
        // reference keyframe of current frame
        KeyFrame* ref_keyframe = curr_frame.mpReferenceKF;
        int pt_idx_in_frame = curr_map_pt->GetIndexInKeyFrame(ref_keyframe);
        if (pt_idx_in_frame >= 0) {
          keypts2_matched_w_prev_.push_back(keypt_curr);
          keypts1_matched_w_curr_.push_back(
              ref_keyframe->mvKeysUn[pt_idx_in_frame]);
          int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
          matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
          err_vals_visualization_.push_back(reproj_err);
        } else {
          LOG(WARNING) << "Map point was not found in current frame's "
                          "reference keyframe.";
        }
      } else {
        // Append it to the keypoint visualization list only if the
        // map point is seen both in current and previous frames.
        if (prev_frame.GetCorrespondingKeyPt(curr_map_pt, &keypt_prev)) {
          keypts2_matched_w_prev_.push_back(keypt_curr);
          keypts1_matched_w_curr_.push_back(keypt_prev);
          int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
          matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
          err_vals_visualization_.push_back(reproj_err);
        }
      }
    }
  }
}

// Calculates the epipolar error rather than the reprojection error for
// keypoints that exist in the current frame
void FeatureEvaluator::EvaluateAgainstPrevFrameEpipolar(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, prev_frame.mTwc_gt);

  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = curr_frame.mvpMapPoints[i];
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;
      int keypt_prev_idx;

      bool prev_keypt_available = prev_frame.GetCorrespondingKeyPt(
          curr_map_pt, &keypt_prev, &keypt_prev_idx);
      if (prev_keypt_available) {
        keypts2_select_.push_back(keypt_curr);
        keypts2_matched_w_prev_.push_back(keypt_curr);
        keypts1_matched_w_curr_.push_back(keypt_prev);
        int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
        matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));

        Vector2f epipolar_line_dir;
        Vector2f proj_on_epipolar_line;
        double err = CalculateEpipolarError(tf_prev_to_curr,
                                            curr_frame,
                                            keypt_prev,
                                            keypt_curr,
                                            &epipolar_line_dir,
                                            &proj_on_epipolar_line);

        cv::Point2f reproj_gt;
        bool gt_depth_available = GetGTReprojection(
            prev_frame, curr_frame, keypt_prev_idx, i, &reproj_gt);
        if (gt_depth_available) {
          reproj2_select_gt_.push_back(reproj_gt);
        }

        err_vals_select_.push_back(err);
        epipolar_lines_dir_.push_back(epipolar_line_dir);
        epipolar_projections_.push_back(proj_on_epipolar_line);
        err_vals_visualization_.push_back(err);
      }
    }
  }
}

// Calculates the epipolar error rather than the reprojection error for
// keypoints that exist in the current frame
void FeatureEvaluator::EvaluateAgainstRefKeyFrameEpipolar(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = curr_frame.mvpMapPoints[i];
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;

      // Reference frame of current map point
      KeyFrame* pt_ref_keyframe = curr_map_pt->GetReferenceKeyFrame();
      //       cout << "Ref frame ID: " << pt_ref_keyframe->mnId << endl;
      //       cout << "curr frame ID: " << curr_frame.mnId << endl;
      int pt_idx_in_frame = curr_map_pt->GetIndexInKeyFrame(pt_ref_keyframe);

      double err;
      Vector2f epipolar_line_dir;
      Vector2f proj_on_epipolar_line;
      if (pt_idx_in_frame >= 0) {
        keypt_prev = pt_ref_keyframe->mvKeysUn[pt_idx_in_frame];
        err = CalculateEpipolarError(*pt_ref_keyframe,
                                     curr_frame,
                                     keypt_prev,
                                     keypt_curr,
                                     &epipolar_line_dir,
                                     &proj_on_epipolar_line);

        cv::Point2f reproj_gt;
        bool uncertain_gt_depth;
        bool gt_depth_available = GetGTReprojection(*pt_ref_keyframe,
                                                    curr_frame,
                                                    pt_idx_in_frame,
                                                    i,
                                                    &reproj_gt,
                                                    &uncertain_gt_depth);
        if (gt_depth_available) {
          if (uncertain_gt_depth) {
            continue;
          } else {
            reproj2_select_gt_.push_back(reproj_gt);
          }
        }

        keypts2_select_.push_back(keypt_curr);
        err_vals_select_.push_back(err);
        epipolar_lines_dir_.push_back(epipolar_line_dir);
        epipolar_projections_.push_back(proj_on_epipolar_line);
      } else {
        LOG(WARNING) << "Could not find a matching keypoint in the "
                        "reference keyframe of current map point!";
        continue;
      }

      // Since for now only the current frame and the previous frame
      // images are available, for visualizing matched keypoints, only
      // show those that are seen in the previous frame as well as
      // current frame.
      if (false) {
        // Find the index of the corresponding descriptor in the reference
        // keyframe of current frame. It is only for visualization purposes.
        // Here keypts1_matched_w_curr_ is filled with keypoints in the
        // reference keyframe of current frame
        KeyFrame* ref_keyframe = curr_frame.mpReferenceKF;
        int pt_idx_in_frame = curr_map_pt->GetIndexInKeyFrame(ref_keyframe);
        if (pt_idx_in_frame >= 0) {
          keypts2_matched_w_prev_.push_back(keypt_curr);
          keypts1_matched_w_curr_.push_back(
              ref_keyframe->mvKeysUn[pt_idx_in_frame]);
          int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
          matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
          err_vals_visualization_.push_back(err);
        } else {
          LOG(WARNING) << "Map point was not found in current frame's "
                          "reference keyframe.";
          continue;
        }
      } else {
        // Append it to the keypoint visualization list only if the
        // map point is seen both in current and previous frames.
        if (prev_frame.GetCorrespondingKeyPt(curr_map_pt, &keypt_prev)) {
          keypts2_matched_w_prev_.push_back(keypt_curr);
          keypts1_matched_w_curr_.push_back(keypt_prev);
          int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
          matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
          err_vals_visualization_.push_back(err);
        }
      }
    }
  }
}

// Calculates the epipolar error rather than the reprojection error for
// keypoints that exist in the current frame. It also uses sigma points
// for calculating the covariance of error and uses that for
// normalization. Only keypoints that are matched with a point in the
// previous frame are used.
void FeatureEvaluator::EvaluateAgainstPrevFrameEpipolarNormalized(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = curr_frame.mvpMapPoints[i];
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;
      int keypt_prev_idx;

      bool prev_keypt_available = prev_frame.GetCorrespondingKeyPt(
          curr_map_pt, &keypt_prev, &keypt_prev_idx);

      if (prev_keypt_available) {
        double err;
        Vector2f epipolar_line_dir;
        Vector2f proj_on_epipolar_line;
        vector<Vector2f> sigma_pts_err;
        Eigen::Matrix2f epipolar_err_covariance;
        double err_norm_factor;
        float epipolar_err_var;

        if (kUseAnalyticalUncertaintyPropagation_) {
          err = CalculateNormalizedEpipolarErrorAnalytical(
              prev_frame.mTwc_gt,
              prev_frame.mSigmaTwc_gt,
              prev_frame.mbPoseUncertaintyAvailable,
              prev_frame.mstrLeftImgName,
              curr_frame,
              keypt_prev,
              keypt_curr,
              &epipolar_line_dir,
              &proj_on_epipolar_line,
              &epipolar_err_var,
              &err_norm_factor);
          // Since in analytical uncertainty propagation mode, we only
          // estimate the variance of the epipolar error magnitude, we fill
          // the epipolar error covariance as a diagonal with epipolar_err_var
          // It is only for qualitative visualization purposes.
          epipolar_err_covariance << epipolar_err_var, 0, 0, epipolar_err_var;

        } else {
          err = CalculateNormalizedEpipolarError(prev_frame.mTwc_gt,
                                                 curr_frame,
                                                 keypt_prev,
                                                 keypt_curr,
                                                 &epipolar_line_dir,
                                                 &proj_on_epipolar_line,
                                                 &sigma_pts_err,
                                                 &epipolar_err_covariance,
                                                 &err_norm_factor);
        }

        // Make sure the normalized error is not a NaN, i.e. the determinent
        // of the covariance matrix is non zeros
        if (isnan(err) || err < 0) {
          LOG(WARNING)
              << "Invalid normalized error value " << err
              << ".Error covariance should have a positive determinant."
              << "The covariance matrix was " << epipolar_err_covariance << endl
              << "determinent: " << epipolar_err_covariance.determinant();
          continue;
        }

        cv::Point2f reproj_gt;
        bool gt_depth_available = GetGTReprojection(
            prev_frame, curr_frame, keypt_prev_idx, i, &reproj_gt);
        if (gt_depth_available) {
          reproj2_select_gt_.push_back(reproj_gt);
        }

        keypts2_select_.push_back(keypt_curr);
        keypts2_select_life_.push_back(curr_frame.mnId -
                                       curr_map_pt->mnFirstFrame);
        keypts2_select_ref_frame_offset_.push_back(
            curr_frame.mnId - curr_map_pt->GetReferenceKeyFrame()->mnFrameId);
        keypts2_matched_w_prev_.push_back(keypt_curr);
        keypts1_matched_w_curr_.push_back(keypt_prev);
        int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
        matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
        err_vals_select_.push_back(err);
        epipolar_lines_dir_.push_back(epipolar_line_dir);
        epipolar_projections_.push_back(proj_on_epipolar_line);
        sigma_pts_err_.push_back(sigma_pts_err);
        err_norm_factor_.push_back(err_norm_factor);
        err_vals_visualization_.push_back(err);
        epi_err_vec_particles_cov_.push_back(epipolar_err_covariance);
      } else {
        LOG(WARNING) << "Could not find a matching keypoint in the "
                        "previous frame!";
      }
    }
  }
}

// Calculates the epipolar error rather than the reprojection error for
// keypoints that exist in the current frame. It also uses sigma points
// for calculating the covariance of error and uses that for
// normalization.
void FeatureEvaluator::EvaluateAgainstRefKeyFrameEpipolarNormalized(
    ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame) {
  // If set to true, all matched keypoints will be evaluated, including those
  // pruned as outliers by ORB-SLAM
  const bool kComprehensiveEvaluation = false;

  // If set to true and ground truth depth images are avaialble, the
  // reprojection error will be calculated and returned instead of the
  // epipolar error
  const bool kOverwriteWithReprojErr = false;

  vector<MapPoint*>* map_pts;
  if (kComprehensiveEvaluation) {
    map_pts = &curr_frame.mvpMapPointsComp;
  } else {
    map_pts = &curr_frame.mvpMapPoints;
  }

  // Number of keypoints in the current frame that are not matched with a
  // map point
  int no_map_point_count = 0;

  // Number of keypoints in the current frame that are matched with a map point
  // However, not with a keypoint from the reference frame of the map point
  int unmatched_keypt_count = 0;
  for (size_t i = 0; i < curr_frame.mvKeysUn.size(); i++) {
    // Add the keypoints that are matched with map points
    MapPoint* curr_map_pt = map_pts->at(i);
    if (curr_map_pt) {
      cv::KeyPoint keypt_curr = curr_frame.mvKeysUn[i];
      cv::KeyPoint keypt_prev;

      // Reference frame of current map point
      KeyFrame* pt_ref_keyframe = curr_map_pt->GetReferenceKeyFrame();
      //       cout << "Ref frame ID: " << pt_ref_keyframe->mnId << endl;
      //       cout << "curr frame ID: " << curr_frame.mnId << endl;
      int pt_idx_in_frame = curr_map_pt->GetIndexInKeyFrame(pt_ref_keyframe);

      double err;
      Vector2f epipolar_line_dir;
      Vector2f proj_on_epipolar_line;
      if (pt_idx_in_frame >= 0) {
        keypt_prev = pt_ref_keyframe->mvKeysUn[pt_idx_in_frame];
        vector<Eigen::Vector2f> sigma_pts_err;
        double err_norm_factor;
        Eigen::Matrix2f epipolar_err_covariance;
        float epipolar_err_var;

        if (kUseAnalyticalUncertaintyPropagation_) {
          err = CalculateNormalizedEpipolarErrorAnalytical(
              pt_ref_keyframe->mTwc_gt,
              pt_ref_keyframe->mSigmaTwc_gt,
              pt_ref_keyframe->mbPoseUncertaintyAvailable,
              pt_ref_keyframe->mstrLeftImgName,
              curr_frame,
              keypt_prev,
              keypt_curr,
              &epipolar_line_dir,
              &proj_on_epipolar_line,
              &epipolar_err_var,
              &err_norm_factor);
          // Since in analytical uncertainty propagation mode, we only
          // estimate the variance of the epipolar error magnitude, we fill
          // the epipolar error covariance as a diagonal with epipolar_err_var
          // It is only for qualitative visualization purposes.
          epipolar_err_covariance << epipolar_err_var, 0, 0, epipolar_err_var;

        } else {
          err = CalculateNormalizedEpipolarError(pt_ref_keyframe->mTwc_gt,
                                                 curr_frame,
                                                 keypt_prev,
                                                 keypt_curr,
                                                 &epipolar_line_dir,
                                                 &proj_on_epipolar_line,
                                                 &sigma_pts_err,
                                                 &epipolar_err_covariance,
                                                 &err_norm_factor);
        }

        // Make sure the normalized error is not a NaN, i.e. the determinent
        // of the covariance matrix is non zeros
        if (isnan(err) || err < 0) {
          LOG(INFO) << "Invalid normalized error value " << err
                    << ".Error covariance should have a positive determinant."
                    << "The covariance matrix was " << epipolar_err_covariance
                    << endl
                    << "determinent: " << epipolar_err_covariance.determinant();
          continue;
        }

        // Do not use keypoints with a track length less than a threshold for
        // training.
        if (kFilterBasedOnTrackLength_) {
          int keypt_life = curr_frame.mnId - curr_map_pt->mnFirstFrame;
          if (keypt_life < kMinTrackLength_) {
            continue;
          }
        }

        cv::Point2f reproj_gt;
        bool uncertain_gt_depth;
        bool gt_depth_available = GetGTReprojection(*pt_ref_keyframe,
                                                    curr_frame,
                                                    pt_idx_in_frame,
                                                    i,
                                                    &reproj_gt,
                                                    &uncertain_gt_depth);
        if (gt_depth_available) {
          if (uncertain_gt_depth) {
            continue;
          } else {
            reproj2_select_gt_.push_back(reproj_gt);

            if (kOverwriteWithReprojErr) {
              err = sqrt((keypt_curr.pt.x - reproj_gt.x) *
                             (keypt_curr.pt.x - reproj_gt.x) +
                         (keypt_curr.pt.y - reproj_gt.y) *
                             (keypt_curr.pt.y - reproj_gt.y));

              // Normalize reprojection error w.r.t. keypoint size
              if (false) {
                float keypt_sigma =
                    1 / curr_frame.mvInvLevelSigma2[keypt_curr.octave];
                float scale95 = sqrt(5.991);
                float normalization_factor =
                    scale95 * static_cast<float>(sqrt(keypt_sigma));
                err = err / normalization_factor;
              }
            }
          }
        }

        // ++++++++++++++++++++++++++++++++++++++++++++++++++++
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++
        // TODO(remove this sanity checking)
        // Add quality score to the keypoints online and given the latest
        // evaluation of each keypoint
        const bool kEnableKeyPointEval = false;
        const bool kUseEpipolarErr = true;
        const float kReprojErrMaxClamp = 100.0;  // 10.0
        const float kMinQualityScore = 0.95;

        if (kEnableKeyPointEval) {
          double scaled_err = 0;
          if (kUseEpipolarErr) {
            double err_log = err;
            scaled_err = static_cast<double>((err_log - kErrMinClamp_) /
                                             (kErrMaxClamp_ - kErrMinClamp_));
            scaled_err = (scaled_err > 0.9) ? 0.9 : scaled_err;
            scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
          } else if (gt_depth_available) {
            // Use ground truth reprojection error
            Vector2f reproj_err(reproj_gt.x - keypt_curr.pt.x,
                                reproj_gt.y - keypt_curr.pt.y);
            float err_abs = sqrt(reproj_err.x() * reproj_err.x() +
                                 reproj_err.y() * reproj_err.y());
            scaled_err = err_abs / kReprojErrMaxClamp;
            scaled_err = (scaled_err > 1.0) ? 1.0 : scaled_err;
            scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
          }

          float qual_score = 1.0 / (1.0 + scaled_err);
          float qual_score_norm = 2 * qual_score - 1;

          curr_map_pt->SetQualityScore(qual_score_norm);
          curr_frame.mvKeyQualScore[i] = qual_score_norm;
        }
        // ------------------------------------------------------
        // ------------------------------------------------------

        keypts2_select_.push_back(keypt_curr);
        keypts2_select_life_.push_back(curr_frame.mnId -
                                       curr_map_pt->mnFirstFrame);
        keypts2_select_ref_frame_offset_.push_back(curr_frame.mnId -
                                                   pt_ref_keyframe->mnFrameId);
        err_vals_select_.push_back(err);
        epipolar_lines_dir_.push_back(epipolar_line_dir);
        epipolar_projections_.push_back(proj_on_epipolar_line);
        err_norm_factor_.push_back(err_norm_factor);
        epi_err_vec_particles_cov_.push_back(epipolar_err_covariance);
        if (!kUseAnalyticalUncertaintyPropagation_) {
          sigma_pts_err_.push_back(sigma_pts_err);
        }
      } else {
        unmatched_keypt_count++;
        //        LOG(INFO) << "Could not find a matching keypoint in the "
        //                      "reference keyframe of current map point!";
        continue;
      }

      // Since for now only the current frame and the previous frame
      // images are available, for visualizing matched keypoints, only
      // show those that are seen in the previous frame as well as
      // current frame.
      if (false) {
        // Find the index of the corresponding descriptor in the reference
        // keyframe of current frame. It is only for visualization purposes.
        // Here keypts1_matched_w_curr_ is filled with keypoints in the
        // reference keyframe of current frame
        KeyFrame* ref_keyframe = curr_frame.mpReferenceKF;
        int pt_idx_in_frame = curr_map_pt->GetIndexInKeyFrame(ref_keyframe);
        if (pt_idx_in_frame >= 0) {
          keypts2_matched_w_prev_.push_back(keypt_curr);
          keypts1_matched_w_curr_.push_back(
              ref_keyframe->mvKeysUn[pt_idx_in_frame]);
          int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
          matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
          err_vals_visualization_.push_back(err);
        } else {
          LOG(WARNING) << "Map point was not found in current frame's "
                          "reference keyframe.";
          continue;
        }
      } else {
        // Append it to the keypoint visualization list only if the
        // map point is seen both in current and previous frames.
        if (prev_frame.GetCorrespondingKeyPt(curr_map_pt, &keypt_prev)) {
          keypts2_matched_w_prev_.push_back(keypt_curr);
          keypts1_matched_w_curr_.push_back(keypt_prev);
          int keypt_idx = (int)(keypts2_matched_w_prev_.size()) - 1;
          matches1to2_.push_back(DMatch(keypt_idx, keypt_idx, 0.0));
          err_vals_visualization_.push_back(err);
        }
      }

    } else {
      no_map_point_count++;
    }
  }

  if (unmatched_keypt_count > 0) {
    LOG(INFO) << "Unmatched keypoints: " << unmatched_keypt_count;
  }

  if (no_map_point_count > 0) {
    LOG(INFO) << "No map point count: " << no_map_point_count << "/ "
              << curr_frame.mvKeysUn.size();
    LOG(INFO) << "Matched to map point count: "
              << curr_frame.mvKeysUn.size() - no_map_point_count;
  }
}

Eigen::Vector2f FeatureEvaluator::CalculateEpipolarErrorVec(
    const Eigen::Matrix3d& R_prev_to_curr,
    const Eigen::Vector3d& trans_prev_to_curr,
    const ORB_SLAM2::Frame& curr_frame,
    const cv::KeyPoint& keypoint1,
    const cv::KeyPoint& keypoint2,
    float* epipolar_err_scalar,
    Eigen::Vector2f* epipolar_line_dir,
    Eigen::Vector2f* proj_on_epipolar_line) {
  Matrix3d t_cross;
  Vector3d trans_prev_to_curr_norm = trans_prev_to_curr.normalized();
  t_cross << 0, -trans_prev_to_curr_norm.z(), trans_prev_to_curr_norm.y(),
      trans_prev_to_curr_norm.z(), 0, -trans_prev_to_curr_norm.x(),
      -trans_prev_to_curr_norm.y(), trans_prev_to_curr_norm.x(), 0;
  Matrix3d essential_mat = t_cross * R_prev_to_curr;

  // Convert the pixel coordinates of the matched points to their normalized
  // 3d point coordinates
  Eigen::Vector3d norm_3d_points1, norm_3d_points2;
  CalculateNormalized3DPoint(
      cv::Point2f(keypoint1.pt.x, keypoint1.pt.y), 0, &norm_3d_points1);
  CalculateNormalized3DPoint(
      cv::Point2f(keypoint2.pt.x, keypoint2.pt.y), 0, &norm_3d_points2);

  // TODO (srabiee): Maybe run the below calculations as an optional part that
  // could be turned off when we do not want to visualize the epipolar error

  // Calculate the epipolar line and the closest point on the line to the given
  // keypoint (keypoint2). In these calculations the epipolar line is calcuated
  // for the current frame and the epipole is the projection of the previous
  // frame's camera center on current frame's image plane
  Vector3f c1_in1(0, 0, 0);
  Vector3f c1_in2_3d =
      R_prev_to_curr.cast<float>() * c1_in1 + trans_prev_to_curr.cast<float>();
  Vector2f epipole = ProjectToCam(curr_frame, c1_in2_3d);

  // The 3d point that was associated to keypoint1 in prev frame, converted to
  // current frame and projected to the image plane
  Vector3f p1_in2_3d =
      R_prev_to_curr.cast<float>() * norm_3d_points1.cast<float>() +
      trans_prev_to_curr.cast<float>();
  Vector2f p1_in2_2d = ProjectToCam(curr_frame, p1_in2_3d);

  // Unit vector of the epipolar line
  Vector2f u_hat = (p1_in2_2d - epipole).normalized();

  // Find the projection of keypoint2 on the epipolar line
  Vector2f keypt_eig(keypoint2.pt.x, keypoint2.pt.y);
  Vector2f epp_proj = epipole + (keypt_eig - epipole).dot(u_hat) * u_hat;

  *epipolar_line_dir = (epp_proj - epipole).normalized();
  *proj_on_epipolar_line = epp_proj;
  // Use sin(theta) as error metric, where theta is the angle between the
  // line connecting the camera center to the matched feature, and the line
  // connecting the camera center to the projection of the feature on epipolar
  // line
  //   *epipolar_err_scalar = err;
  // Use the length of the distance to epipolar line (in pixels) as the
  // error metric
  *epipolar_err_scalar = (keypt_eig - epp_proj).norm();
  //   *epipolar_err_scalar = *epipolar_err_scalar /
  //                           (projection_mat_cam0_(0,0));
  return epp_proj - keypt_eig;
}

double FeatureEvaluator::CalculateEpipolarError(
    const ORB_SLAM2::KeyFrame& ref_keyframe,
    const ORB_SLAM2::Frame& curr_frame,
    const cv::KeyPoint& keypoint1,
    const cv::KeyPoint& keypoint2,
    Eigen::Vector2f* epipolar_line_dir,
    Eigen::Vector2f* proj_on_epipolar_line) {
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, ref_keyframe.mTwc_gt);
  Vector3d trans_prev_to_curr(double(tf_prev_to_curr.at<float>(0, 3)),
                              double(tf_prev_to_curr.at<float>(1, 3)),
                              double(tf_prev_to_curr.at<float>(2, 3)));
  Matrix3d R_prev_to_curr;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      R_prev_to_curr(i, j) = double(tf_prev_to_curr.at<float>(i, j));
    }
  }

  Matrix3d t_cross;
  Vector3d trans_prev_to_curr_norm = trans_prev_to_curr.normalized();
  t_cross << 0, -trans_prev_to_curr_norm.z(), trans_prev_to_curr_norm.y(),
      trans_prev_to_curr_norm.z(), 0, -trans_prev_to_curr_norm.x(),
      -trans_prev_to_curr_norm.y(), trans_prev_to_curr_norm.x(), 0;
  Matrix3d essential_mat = t_cross * R_prev_to_curr;

  // Convert the pixel coordinates of the matched points to their normalized
  // 3d point coordinates
  Eigen::Vector3d norm_3d_points1, norm_3d_points2;
  CalculateNormalized3DPoint(
      cv::Point2f(keypoint1.pt.x, keypoint1.pt.y), 0, &norm_3d_points1);
  CalculateNormalized3DPoint(
      cv::Point2f(keypoint2.pt.x, keypoint2.pt.y), 0, &norm_3d_points2);

  double err =
      fabs(norm_3d_points2.transpose() * essential_mat * norm_3d_points1);

  // TODO (srabiee): Maybe run the below calculations as an optional part that
  // could be turned off when we do not want to visualize the epipolar error

  // Calculate the epipolar line and the closest point on the line to the given
  // keypoint (keypoint2). In these calculations the epipolar line is calcuated
  // for the current frame and the epipole is the projection of the previous
  // frame's camera center on current frame's image plane
  Vector3f c1_in1(0, 0, 0);
  Vector3f c1_in2_3d =
      R_prev_to_curr.cast<float>() * c1_in1 + trans_prev_to_curr.cast<float>();
  Vector2f epipole = ProjectToCam(curr_frame, c1_in2_3d);

  // The 3d point that was associated to keypoint1 in prev frame, converted to
  // current frame and projected to the image plane
  Vector3f p1_in2_3d =
      R_prev_to_curr.cast<float>() * norm_3d_points1.cast<float>() +
      trans_prev_to_curr.cast<float>();
  Vector2f p1_in2_2d = ProjectToCam(curr_frame, p1_in2_3d);

  // Unit vector of the epipolar line
  Vector2f u_hat = (p1_in2_2d - epipole).normalized();

  // Find the projection of keypoint2 on the epipolar line
  Vector2f keypt_eig(keypoint2.pt.x, keypoint2.pt.y);
  Vector2f epp_proj = epipole + (keypt_eig - epipole).dot(u_hat) * u_hat;

  *epipolar_line_dir = (epp_proj - epipole).normalized();
  *proj_on_epipolar_line = epp_proj;
  return err;
}

double FeatureEvaluator::CalculateEpipolarError(
    const cv::Mat& tf_prev_to_curr,
    const ORB_SLAM2::Frame& curr_frame,
    const cv::KeyPoint& keypoint1,
    const cv::KeyPoint& keypoint2,
    Eigen::Vector2f* epipolar_line_dir,
    Eigen::Vector2f* proj_on_epipolar_line) {
  Vector3d trans_prev_to_curr(double(tf_prev_to_curr.at<float>(0, 3)),
                              double(tf_prev_to_curr.at<float>(1, 3)),
                              double(tf_prev_to_curr.at<float>(2, 3)));
  Matrix3d R_prev_to_curr;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      R_prev_to_curr(i, j) = double(tf_prev_to_curr.at<float>(i, j));
    }
  }

  Matrix3d t_cross;
  Vector3d trans_prev_to_curr_norm = trans_prev_to_curr.normalized();
  t_cross << 0, -trans_prev_to_curr_norm.z(), trans_prev_to_curr_norm.y(),
      trans_prev_to_curr_norm.z(), 0, -trans_prev_to_curr_norm.x(),
      -trans_prev_to_curr_norm.y(), trans_prev_to_curr_norm.x(), 0;
  Matrix3d essential_mat = t_cross * R_prev_to_curr;

  // Convert the pixel coordinates of the matched points to their normalized
  // 3d point coordinates
  Eigen::Vector3d norm_3d_points1, norm_3d_points2;
  CalculateNormalized3DPoint(
      cv::Point2f(keypoint1.pt.x, keypoint1.pt.y), 0, &norm_3d_points1);
  CalculateNormalized3DPoint(
      cv::Point2f(keypoint2.pt.x, keypoint2.pt.y), 0, &norm_3d_points2);

  double err =
      fabs(norm_3d_points2.transpose() * essential_mat * norm_3d_points1);

  // TODO (srabiee): Maybe run the below calculations as an optional part that
  // could be turned off when we do not want to visualize the epipolar error

  // Calculate the epipolar line and the closest point on the line to the given
  // keypoint (keypoint2). In these calculations the epipolar line is calcuated
  // for the current frame and the epipole is the projection of the previous
  // frame's camera center on current frame's image plane
  Vector3f c1_in1(0, 0, 0);
  Vector3f c1_in2_3d =
      R_prev_to_curr.cast<float>() * c1_in1 + trans_prev_to_curr.cast<float>();
  Vector2f epipole = ProjectToCam(curr_frame, c1_in2_3d);

  // The 3d point that was associated to keypoint1 in prev frame, converted to
  // current frame and projected to the image plane
  Vector3f p1_in2_3d =
      R_prev_to_curr.cast<float>() * norm_3d_points1.cast<float>() +
      trans_prev_to_curr.cast<float>();
  Vector2f p1_in2_2d = ProjectToCam(curr_frame, p1_in2_3d);

  // Unit vector of the epipolar line
  Vector2f u_hat = (p1_in2_2d - epipole).normalized();

  // Find the projection of keypoint2 on the epipolar line
  Vector2f keypt_eig(keypoint2.pt.x, keypoint2.pt.y);
  Vector2f epp_proj = epipole + (keypt_eig - epipole).dot(u_hat) * u_hat;

  *epipolar_line_dir = (epp_proj - epipole).normalized();
  *proj_on_epipolar_line = epp_proj;

  return err;
}

void FeatureEvaluator::CalculateNormalized3DPoint(
    const cv::Point2f& pixel_coord,
    const int cam_id,
    Eigen::Vector3d* normalized_3d_coord) {
  double fx, fy, px, py;

  if (cam_id == 0) {
    fx = projection_mat_cam0_(0, 0);
    fy = projection_mat_cam0_(1, 1);
    px = projection_mat_cam0_(0, 2);
    py = projection_mat_cam0_(1, 2);
  } else if (cam_id == 1) {
    fx = projection_mat_cam1_(0, 0);
    fy = projection_mat_cam1_(1, 1);
    px = projection_mat_cam1_(0, 2);
    py = projection_mat_cam1_(1, 2);
  } else {
    LOG(FATAL) << "Unknown camera id " << cam_id;
  }

  double x = (pixel_coord.x - px) / fx;
  double y = (pixel_coord.y - py) / fy;
  *normalized_3d_coord = Vector3d(x, y, 1.0);
}

double FeatureEvaluator::CalculateReprojectionError(
    const ORB_SLAM2::Frame& ref_frame,
    const ORB_SLAM2::Frame& curr_frame,
    const cv::Mat& tf_prev_to_curr,
    ORB_SLAM2::MapPoint& map_point,
    const cv::KeyPoint& keypoint,
    cv::Point2f* reproj_pt) {
  // Map point's ground truth 3d location in current frame's coordinate.
  // It is achieved by applying the ground truth transformation from ref_frame
  // to curr_frame (T_ref_curr) to the map point's location in the coordinate
  // of the ref_frame
  cv::Mat X_pt_curr(4, 1, CV_32F);
  cv::Mat map_point_w = cv::Mat::ones(4, 1, CV_32F);
  // Convert the map point position to the homogeneous coordinate
  map_point.GetWorldPos().copyTo(map_point_w.rowRange(0, 3));

  X_pt_curr = tf_prev_to_curr * ref_frame.mTcw * map_point_w;

  cv::Mat u_pt_curr = ProjectToCam(curr_frame, X_pt_curr.rowRange(0, 3));
  reproj_pt->x = u_pt_curr.at<float>(0, 0);
  reproj_pt->y = u_pt_curr.at<float>(0, 1);

  float delta_x = u_pt_curr.at<float>(0, 0) - keypoint.pt.x;
  float delta_y = u_pt_curr.at<float>(0, 1) - keypoint.pt.y;
  float reprojection_err = sqrt(delta_x * delta_x + delta_y * delta_y);

  return reprojection_err;
}

double FeatureEvaluator::CalculateReprojectionError(
    const ORB_SLAM2::Frame& ref_frame,
    const ORB_SLAM2::Frame& curr_frame,
    ORB_SLAM2::MapPoint& map_point,
    const cv::KeyPoint& keypoint,
    cv::Point2f* reproj_pt) {
  // Map point's ground truth 3d location in current frame's coordinate.
  // It is achieved by applying the ground truth transformation from ref_frame
  // to curr_frame (T_ref_curr) to the map point's location in the coordinate
  // of the ref_frame
  cv::Mat X_pt_curr(4, 1, CV_32F);
  cv::Mat map_point_w = cv::Mat::ones(4, 1, CV_32F);
  // Convert the map point position to the homogeneous coordinate
  map_point.GetWorldPos().copyTo(map_point_w.rowRange(0, 3));
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, ref_frame.mTwc_gt);
  float scale = 1.0;
  tf_prev_to_curr.rowRange(0, 3).col(3) =
      tf_prev_to_curr.rowRange(0, 3).col(3) / scale;

  X_pt_curr = tf_prev_to_curr * ref_frame.mTcw * map_point_w;

  cv::Mat u_pt_curr = ProjectToCam(curr_frame, X_pt_curr.rowRange(0, 3));
  reproj_pt->x = u_pt_curr.at<float>(0, 0);
  reproj_pt->y = u_pt_curr.at<float>(0, 1);

  float delta_x = u_pt_curr.at<float>(0, 0) - keypoint.pt.x;
  float delta_y = u_pt_curr.at<float>(0, 1) - keypoint.pt.y;
  float reprojection_err = sqrt(delta_x * delta_x + delta_y * delta_y);

  return reprojection_err;
}

double FeatureEvaluator::CalculateReprojectionError(
    ORB_SLAM2::KeyFrame& ref_frame,
    const ORB_SLAM2::Frame& curr_frame,
    ORB_SLAM2::MapPoint& map_point,
    const cv::KeyPoint& keypoint,
    cv::Point2f* reproj_pt) {
  // Map point's ground truth 3d location in current frame's coordinate.
  // It is achieved by applying the ground truth transformation from ref_frame
  // to curr_frame (T_ref_curr) to the map point's location in the coordinate
  // of the ref_frame
  cv::Mat X_pt_curr(4, 1, CV_32F);
  cv::Mat map_point_w = cv::Mat::ones(4, 1, CV_32F);
  // Convert the map point position to the homogeneous coordinate
  map_point.GetWorldPos().copyTo(map_point_w.rowRange(0, 3));
  cv::Mat tf_prev_to_curr =
      CalculateRelativeTransform(curr_frame.mTwc_gt, ref_frame.mTwc_gt);
  float scale = 1.0;
  tf_prev_to_curr.rowRange(0, 3).col(3) =
      tf_prev_to_curr.rowRange(0, 3).col(3) / scale;

  X_pt_curr = tf_prev_to_curr * ref_frame.GetPose() * map_point_w;

  cv::Mat u_pt_curr = ProjectToCam(curr_frame, X_pt_curr.rowRange(0, 3));
  reproj_pt->x = u_pt_curr.at<float>(0, 0);
  reproj_pt->y = u_pt_curr.at<float>(0, 1);

  float delta_x = u_pt_curr.at<float>(0, 0) - keypoint.pt.x;
  float delta_y = u_pt_curr.at<float>(0, 1) - keypoint.pt.y;
  float reprojection_err = sqrt(delta_x * delta_x + delta_y * delta_y);

  return reprojection_err;
}

cv::Mat FeatureEvaluator::GenerateErrHeatmap(
    unsigned int rows, unsigned int cols, const std::vector<double> err_vals) {
  const double err_max_clamp = kErrMaxClamp_;
  const double err_min_clamp = kErrMinClamp_;

  double pv[err_vals.size()];
  for (unsigned int i = 0; i < err_vals.size(); i++) {
    double scaled_err = static_cast<double>((err_vals[i] - err_min_clamp) /
                                            (err_max_clamp - err_min_clamp));
    scaled_err = (scaled_err > 1.0) ? 1.0 : scaled_err;
    scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
    pv[i] = scaled_err;
  }

  cv::Mat out_img(rows, cols, CV_64FC1);
  memcpy(out_img.data, &pv, err_vals.size() * sizeof(err_vals[0]));

  return out_img;
}

cv::Mat FeatureEvaluator::GenerateErrHeatmap(unsigned int rows,
                                             unsigned int cols,
                                             const std::vector<double> err_vals,
                                             double err_max_clamp,
                                             double err_min_clamp) {
  double pv[err_vals.size()];
  for (unsigned int i = 0; i < err_vals.size(); i++) {
    double scaled_err = static_cast<double>((err_vals[i] - err_min_clamp) /
                                            (err_max_clamp - err_min_clamp));
    scaled_err = (scaled_err > 1.0) ? 1.0 : scaled_err;
    scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
    pv[i] = scaled_err;
  }

  cv::Mat out_img(rows, cols, CV_64FC1);
  memcpy(out_img.data, &pv, err_vals.size() * sizeof(err_vals[0]));

  return out_img;
}

void FeatureEvaluator::Hist2D(const std::vector<double>& x_in,
                              const std::vector<double>& y_in,
                              const std::vector<float>& values,
                              double bin_size_x,
                              double bin_size_y,
                              double stride_x,
                              double stride_y,
                              int bin_num_x,
                              int bin_num_y,
                              Eigen::ArrayXXd* freq,
                              Eigen::ArrayXXd* bin_val) {
  // When calculating the value of each bin, do a weighted mean where the
  // weights are calculated using a kernel placed on the center of the bin
  const bool kWeightUsingKernel = false;

  // Go through the datapoints and put them in the corresponding bins
  for (size_t i = 0; i < values.size(); i++) {
    // Each keypoint could fall into multiple bins as they are overlapping
    vector<int> idx_x_vec;
    vector<int> idx_y_vec;

    int idx_x_st = std::ceil(x_in[i] / stride_x - bin_size_x / stride_x);
    int idx_y_st = std::ceil(y_in[i] / stride_y - bin_size_y / stride_y);
    int idx_x_end = std::floor(x_in[i] / stride_x);
    int idx_y_end = std::floor(y_in[i] / stride_y);

    int idx_x = idx_x_st;
    int idx_y = idx_y_st;
    while (idx_x <= idx_x_end) {
      if (idx_x >= 0) {
        idx_x_vec.push_back(idx_x);
      }
      idx_x++;
    }

    while (idx_y <= idx_y_end) {
      if (idx_y >= 0) {
        idx_y_vec.push_back(idx_y);
      }
      idx_y++;
    }

    for (const int& idx_x : idx_x_vec) {
      for (const int& idx_y : idx_y_vec) {
        // Crop out the stips at the right and bottom of the image that are
        // not fully tiled with the bins
        if (idx_x >= bin_num_x || idx_y >= bin_num_y) {
          continue;
        }

        if (!kWeightUsingKernel) {
          bin_val->coeffRef(idx_x, idx_y) += static_cast<double>(values[i]);
          freq->coeffRef(idx_x, idx_y) += 1;
        } else {
          Vector2f bin_ctr(idx_x * stride_x + bin_size_x / 2.0,
                           idx_y * stride_y + bin_size_y / 2.0);
          double weight = Kernel(Vector2f(x_in[i], y_in[i]),
                                 bin_ctr,
                                 1.0,
                                 (bin_size_x + bin_size_y) / 8.0);
          bin_val->coeffRef(idx_x, idx_y) +=
              weight * static_cast<double>(values[i]);
          freq->coeffRef(idx_x, idx_y) += weight;
        }
      }
    }
  }
}

Eigen::ArrayXXd FeatureEvaluator::ClampArray(const Eigen::ArrayXXd& input_array,
                                             int max_value) {
  Eigen::ArrayXXd clamped_array(input_array);
  for (int i = 0; i < input_array.size(); i++) {
    clamped_array(i) =
        (input_array(i) > max_value) ? max_value : input_array(i);
  }

  return clamped_array;
}

cv::Mat FeatureEvaluator::OverlayHeatmapOnImage(const cv::Mat& heatmap) {
  // Crop the original image to the size of the heatmap
  cv::Rect heatmap_roi(0, 0, heatmap.cols, heatmap.rows);

  // Use the image that is annotated with keypoint locations as background
  cv::Mat cropped_image = img_feature_qual_annotation_(heatmap_roi);

  if (cropped_image.size() != heatmap.size() ||
      cropped_image.type() != heatmap.type()) {
    LOG(ERROR) << "The images to be overlaid should have the same "
                  "type and size."
               << endl;
  }

  cv::Mat merged_image;
  addWeighted(cropped_image, 0.5, heatmap, 0.5, 0.0, merged_image);

  return merged_image;
}

cv::Mat FeatureEvaluator::CalculateRelativeTransform(
    const cv::Mat& dest_frame_pose, const cv::Mat& src_frame_pose) {
  return CalculateInverseTransform(dest_frame_pose) * src_frame_pose;
}

cv::Mat FeatureEvaluator::CalculateRelativeTransform(
    const cv::Mat& dest_frame_pose,
    const cv::Mat& src_frame_pose,
    const Eigen::Matrix<double, 6, 6>& dest_frame_pose_cov,
    const Eigen::Matrix<double, 6, 6>& src_frame_pose_cov,
    Eigen::Matrix<double, 6, 6>* rel_pos_cov) {
  cv::Mat dest_fram_pose_inv = CalculateInverseTransform(dest_frame_pose);

  // Compute the covariance of the relative transform

  // First, compute the terms that are required for generating the Jacobians
  // terms:

  // Inverse of rotation part of dest_frame_pose
  Eigen::Matrix3d R2_inv;
  Eigen::Vector3d t2_inv;
  Eigen::Vector3d t1;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R2_inv(i, j) = static_cast<double>(dest_fram_pose_inv.at<float>(i, j));
    }
  }

  for (int i = 0; i < 3; i++) {
    t2_inv(i) = static_cast<double>(dest_fram_pose_inv.at<float>(i, 3));
  }

  for (int i = 0; i < 3; i++) {
    t1(i) = static_cast<double>(src_frame_pose.at<float>(i, 3));
  }

  // The Jacobian of the translational part of tf_dest_src w.r.t. the
  // translational part of tf_src (t1)
  Eigen::Matrix3d J_t_t1 = R2_inv;

  // The Jacobian of the translational part of tf_dest_src w.r.t. the
  // translational part of tf_dest (t2)
  Eigen::Matrix3d J_t_t2 = -R2_inv;

  // The Jacobian of the translational part of tf_dest_src w.r.t. the
  // rotational part of tf_dest (w2)
  Eigen::Matrix3d J_t_w2 = GetSkewSymmetric(R2_inv * t1 + t2_inv) * R2_inv;

  // The Jacobian of the rotational part of tf_dest_src w.r.t. the
  // rotational part of tf_src (w1)
  Eigen::Matrix3d J_w_w1 = R2_inv;

  // The Jacobian of the rotational part of tf_dest_src w.r.t. the
  // rotational part of tf_dest (w2)
  Eigen::Matrix3d J_w_w2 = -R2_inv;

  // Covariance of the rotational part of tf_dest_src
  Eigen::Matrix3d Sigma_w;
  Sigma_w =
      J_w_w1 * src_frame_pose_cov.topLeftCorner(3, 3) * J_w_w1.transpose() +
      J_w_w2 * dest_frame_pose_cov.topLeftCorner(3, 3) * J_w_w2.transpose();

  // Covariance of the translational part of tf_dest_src
  Eigen::Matrix3d Sigma_t;
  Sigma_t =
      J_t_t1 * src_frame_pose_cov.bottomRightCorner(3, 3) * J_t_t1.transpose() +
      J_t_t2 * dest_frame_pose_cov.bottomRightCorner(3, 3) *
          J_t_t2.transpose() +
      J_t_w2 * dest_frame_pose_cov.topLeftCorner(3, 3) * J_t_w2.transpose();

  rel_pos_cov->setZero();
  rel_pos_cov->topLeftCorner(3, 3) = Sigma_w;
  rel_pos_cov->bottomRightCorner(3, 3) = Sigma_t;

  return dest_fram_pose_inv * src_frame_pose;
}

cv::Mat FeatureEvaluator::CalculateInverseTransform(const cv::Mat& transform) {
  cv::Mat R1 = transform.rowRange(0, 3).colRange(0, 3);
  cv::Mat t1 = transform.rowRange(0, 3).col(3);
  cv::Mat R1_inv = R1.t();
  cv::Mat t1_inv = -R1_inv * t1;
  cv::Mat transform_inv = cv::Mat::eye(4, 4, transform.type());

  R1_inv.copyTo(transform_inv.rowRange(0, 3).colRange(0, 3));
  t1_inv.copyTo(transform_inv.rowRange(0, 3).col(3));

  return transform_inv;
}

cv::Mat FeatureEvaluator::ProjectToCam(const ORB_SLAM2::Frame& cam_frame,
                                       const cv::Mat& point_3d_in_cam_ref) {
  CHECK_EQ(point_3d_in_cam_ref.size().height, 3);
  CHECK_EQ(point_3d_in_cam_ref.size().width, 1);

  cv::Mat cam_mat = (cv::Mat_<float>(3, 3) << cam_frame.fx,
                     0.0,
                     cam_frame.cx,
                     0.0,
                     cam_frame.fy,
                     cam_frame.cy,
                     0.0,
                     0.0,
                     1.0);

  if (point_3d_in_cam_ref.at<float>(2, 0) == 0) {
    LOG(INFO) << "The provided map point has a depth of 0!";
  }
  return (cam_mat * point_3d_in_cam_ref) / point_3d_in_cam_ref.at<float>(2, 0);
}

Eigen::Vector2f FeatureEvaluator::ProjectToCam(
    const ORB_SLAM2::Frame& cam_frame,
    const Eigen::Vector3f& point_3d_in_cam_ref) {
  CHECK_EQ(point_3d_in_cam_ref.size(), 3);

  Eigen::Matrix3f cam_mat;
  cam_mat << cam_frame.fx, 0.0, cam_frame.cx, 0.0, cam_frame.fy, cam_frame.cy,
      0.0, 0.0, 1.0;

  if (point_3d_in_cam_ref(2) == 0) {
    LOG(INFO) << "The provided map point has a depth of 0!";
  }
  Vector3f projection =
      (cam_mat * point_3d_in_cam_ref) / point_3d_in_cam_ref(2);
  return projection.head(2);
}

cv::Mat FeatureEvaluator::UnrectifyImage(const cv::Mat& input_img) {
  if (rectification_map_available_) {
    cv::Mat unrect_img;
    cv::remap(input_img,
              unrect_img,
              unrect_map1_left_,
              unrect_map2_left_,
              cv::INTER_LINEAR);
    return unrect_img;
  } else {
    return input_img;
  }
}

}  // namespace feature_evaluation
