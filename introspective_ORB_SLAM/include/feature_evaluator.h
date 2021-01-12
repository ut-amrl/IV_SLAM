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
#ifndef IVSLAM_FEATURE_EVALUATOR
#define IVSLAM_FEATURE_EVALUATOR

#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "io_access.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
// #include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"

namespace feature_evaluation {

typedef Eigen::Matrix<float, 6, 1> Vector6f;

enum DescriptorType { kFAST, kORB, kSIFT, kSURF };

enum FeatureMatchingType { kBruteForce, kFLANN, kOpticalFlow };

enum Dataset { kEuroc, kKITTI };

// In the first training mode, the ground truth relative
// transformation is calculated with respect to previous frame and map points
// that do not have a correspondence will be ignored.
// In the second and third training modes, map points that do not have a
// correspondence with previous frame, will be evaluated against the last
// keyframe and their reference frame respectively.
enum TrainingMode {
  kCompareAgainstPrevFrame,
  kCompareAgainstPrevFrameAndLastKeyFrame,
  kCompareAgainstPrevFrameAndRefKeyFrame,
  kCompareAgainstLastKeyFrame,
  kCompareAgainstRefKeyFrame,
  kCompareAgainstPrevFrameEpipolar,
  kCompareAgainstRefKeyFrameEpipolar,
  kCompareAgainstPrevFrameEpipolarNormalized,
  kCompareAgainstRefKeyFrameEpipolarNormalized
};

enum ErrorType { kReprojection, kEpipolar };

// In the first visualization mode, all keypoints that are used for training
// are visualized. (For visualizing matched features across two images
// only those that were matched with the reference frame in current training
// mode are visualized)
// In the second visualization mode, only keypoints that have matches in the
// current training mode's reference frame are used. It could be either the
// the previous frame or the key reference frame.
enum VisualizationMode { kVisualizeAll, kVisualizeOnlyMatchedWithRef };

enum Reliability { Reliable, Unreliable, Unknown };

class FeatureEvaluator {
 public:
  FeatureEvaluator(DescriptorType descriptor_type, Dataset dataset);

  FeatureEvaluator() = default;

  ~FeatureEvaluator() = default;

  void LoadImagePair(cv::Mat img_prev, cv::Mat img_curr);

  void EvaluateFeatures(ORB_SLAM2::Frame& prev_frame,
                        ORB_SLAM2::Frame& curr_frame);

  // Updates the camera calibration for the specified camera using the
  // calibration information stored in the input frame
  void UpdateCameraCalibration(const ORB_SLAM2::Frame& frame);

  // Reads the calibration file and creates the inverse rectification map.
  // It is used to unrectify the generated heatmaps so that they match the
  // original image before rectification
  void LoadRectificationMap(const std::string& calib_file);

  std::vector<cv::KeyPoint> GetMatchedKeyPoints();

  cv::Mat GetAnnotatedImg();

  cv::Mat GetBadMatchAnnotatedImg();

  cv::Mat GetUndistortedImg();

  cv::Mat GetFeatureErrVisualization();

  cv::Mat ColorKeypoints(cv::Mat& img,
                         const std::vector<cv::KeyPoint>& keypts,
                         const std::vector<double>& scalars,
                         double scalar_max_clamp,
                         double drawing_radius);

  cv::Mat GetBadRegionHeatmap();

  cv::Mat GetBadRegionHeatmapMask();

  std::vector<float> GetErrValues();

  // Prepares training images of image quality for SLAM/VO
  void GenerateImageQualityHeatmap();

  // Prepares training images of image quality for SLAM/VO. This version
  // generates a heatmap purely based on the unsupervised map point quality
  // values estimated by outlier analysis
  void GenerateUnsupImageQualityHeatmap(ORB_SLAM2::Frame& frame,
                                        const std::string target_path);

  // Prepares training images of image quality for SLAM/VO. This version uses
  // Gaussian process for interpolating image quality values instead of the
  // sliding window approach offered by GenerateImageQualityHeatmap()
  void GenerateImageQualityHeatmapGP();

  // Prepares training images of image quality for SLAM/VO using Gaussian
  // process for interpolating image quality values. This version generates a
  // heatmap purely based on the unsupervised map point quality values
  // estimated by outlier analysis
  void GenerateUnsupImageQualityHeatmapGP(ORB_SLAM2::Frame& frame);

  // Visualizes reprojection error on current image via drawing arrows
  // from map point reprojection to the corresponding keypoints. Returns false
  // if the active error_type_ is not reprojection error.
  bool DrawReprojectionErrVec();

  // Visualizes the epipolar error on the current image via drawing arrows
  // from the keypoints to their projection on their corresponding epipolar
  // line. Returns false if the active error_type_ is not epipolar error.
  bool DrawEpipolarErrVec(bool was_recently_reset = false);

  void SaveImagesToFile(std::string target_path,
                        const std::string& img_name,
                        bool was_recently_reset = false);

  // Calculates the ground truth 3D point corresponding to a key point in the
  // reference frame and reprojects that to current frame. If ground truth
  // depth is not available for the reference frame returns false.
  bool GetGTReprojection(const ORB_SLAM2::Frame& ref_frame,
                         const ORB_SLAM2::Frame& curr_frame,
                         const int& keypt_idx_in_ref,
                         const size_t& keypt_idx_in_curr,
                         cv::Point2f* reprojection_pt);

  // Calculates the ground truth 3D point corresponding to a key point in the
  // reference KeyFrame and reprojects that to current frame. If ground truth
  // depth is not available for the reference frame returns false.
  bool GetGTReprojection(const ORB_SLAM2::KeyFrame& ref_keyframe,
                         const ORB_SLAM2::Frame& curr_frame,
                         const int& keypt_idx_in_ref,
                         const size_t& keypt_idx_in_curr,
                         cv::Point2f* reprojection_pt,
                         bool* uncertain_gt_depth);

  // Calculates the epipolar error for a pair of keypoints given their
  // corresponding frame (which includes the ground truth pose of camera)
  // It normalizes the resultant epipolar error given the error covariance
  // of a set of sigma points
  double CalculateNormalizedEpipolarError(
      const cv::Mat& ref_frame_Twc_gt,
      const ORB_SLAM2::Frame& curr_frame,
      const cv::KeyPoint& keypoint1,
      const cv::KeyPoint& keypoint2,
      Eigen::Vector2f* epipolar_line_dir,
      Eigen::Vector2f* proj_on_epipolar_line,
      std::vector<Eigen::Vector2f>* sigma_pts_err,
      Eigen::Matrix2f* err_covariance,
      double* err_norm_factor);

  // Calculates the epipolar error for a pair of keypoints given their
  // corresponding frame (which includes the ground truth pose of camera)
  // It normalizes the resultant epipolar error given the covariance of epipolar
  // error for the specific point calculated analytically by propagating the
  // uncertainty in
  // the reference pose of the camera (the uncertainty in the transformation
  // from ref frame to current frame)
  double CalculateNormalizedEpipolarErrorAnalytical(
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
      double* err_norm_factor);

  // Calculates the jacobians of the "epipolar error" (the scalar value)
  // associated with the provided
  // point in the reference frame as well as the relative transformation from
  // ref frame to current frame. J_w is the jacobian w.r.t. perturbations in
  // rotational component of transformation. J_t is the jacobian w.r.t.
  // to the translational part
  void GetEpipolarErrorJacobians(const ORB_SLAM2::Frame& cam_frame,
                                 const Eigen::Matrix3d& R_prev_to_curr,
                                 const Eigen::Vector3d& trans_prev_to_curr,
                                 const Eigen::Vector3d& x_ref,
                                 const Eigen::Vector3d& x,
                                 Eigen::RowVector3d* J_w,
                                 Eigen::RowVector3d* J_t);

  // Calculates the jacobians of the "epipolar line" (represented as a 3D
  // vector in homogenous coordinate) associated with the provided
  // point in the reference frame as well as the relative transformation from
  // ref frame to current frame. J_w is the jacobian w.r.t. perturbations in
  // rotational component of transformation. J_t is the jacobian w.r.t.
  // to the translational part
  void GetEpipolarLineJacobians(const Eigen::Matrix3d& cam_mat,
                                const Eigen::Matrix3d& R_prev_to_curr,
                                const Eigen::Vector3d& trans_prev_to_curr,
                                const Eigen::Vector3d& x_ref,
                                Eigen::Matrix3d* J_w,
                                Eigen::Matrix3d* J_t);

  Eigen::Matrix3d GetFundamentalMatrix(
      const Eigen::Matrix3d& cam_mat,
      const Eigen::Matrix3d& R_prev_to_curr,
      const Eigen::Vector3d& trans_prev_to_curr);

  // Returns the skew symmetric matrix for the input vector
  Eigen::Matrix3d GetSkewSymmetric(Eigen::Vector3d vec);

  // Generates the KMatrix used for gaussian process
  Eigen::MatrixXf Kmatrix(const std::vector<Eigen::Vector2f>& X);

  // Gaussian process prediction
  void GPPredict(float x,
                 float y,
                 const std::vector<Eigen::Vector2f>& locs,
                 const Eigen::VectorXf& values,
                 const Eigen::MatrixXf& K_mat,
                 float& mean,
                 float& variance);

  // Returns true if the latest frame is considered useful for training the
  // introspection model given the percentage of bad matches.
  bool IsFrameGoodForTraining();

  // Used for flagging current frame as either reliable or unreliable for
  // training
  void SetFrameReliability(Reliability frame_reliability);

  // Set the list of relative camera pose uncertainty for all frames. This
  // should only be called in training mode and if such information is
  // available
  void SetRelativeCamPoseUncertainty(
      const std::unordered_map<std::string, int>* pose_unc_map,
      const std::vector<Eigen::Vector2f>* rel_cam_poses_uncertainty);

  bool GetRelativePoseUncertainty(const std::string& ref_frame_name,
                                  const std::string& curr_frame_name,
                                  Eigen::Matrix<double, 6, 6>* rel_pos_cov);

  void CalcRelativePoseError(cv::Mat& pose_est_0,
                             cv::Mat& pose_est_1,
                             cv::Mat& pose_gt_0,
                             cv::Mat& pose_gt_1,
                             Eigen::AngleAxisd* aa_rot_err,
                             Eigen::Vector3d* t_err);

  // Returns the transformation from src_frame to dest_frame (takes a point from
  // src_frame to dest_frame)
  cv::Mat CalculateRelativeTransform(const cv::Mat& dest_frame_pose,
                                     const cv::Mat& src_frame_pose);

  // Returns the transformation from src_frame to dest_frame (takes a point from
  // src_frame to dest_frame). It also computes the covariance of the relative
  // transformation given the covariance of both frames
  cv::Mat CalculateRelativeTransform(
      const cv::Mat& dest_frame_pose,
      const cv::Mat& src_frame_pose,
      const Eigen::Matrix<double, 6, 6>& dest_frame_pose_cov,
      const Eigen::Matrix<double, 6, 6>& src_frame_pose_cov,
      Eigen::Matrix<double, 6, 6>* rel_pos_cov);

  cv::Mat CalculateInverseTransform(const cv::Mat& transform);

 private:
  // Epipolar error threshold for defining bad feature matches (for
  // visualization only)
  const float kBadFeatureErrThresh_Epipolar_ = 1.0;  // Jackal: 0.7
                                                     // Kitti: 0.05,
                                                     // Euroc: 0.35
                                                     // Kitti: 0.005
                                                     // logScale: -7
                                                     // 0.4

  // Reprojection error threshold for defining bad feature matches (for
  // visualization only)
  const float kBadFeatureErrThresh_Reproj_ = 5.0;

  // Parameters for scaling the error values (for both visualization and
  // normalization of training data)
  const double kErrMinClamp_ = 0.0;  // -6.13
  const double kErrMaxClamp_ = 1.5;  // 2 -7  3.87

  // Values to be used as a representative error value for good and bad
  // features when we deal with them in a binary classification manner.
  const float kBadFeatureErrVal_ = 1.5;  // 2
                                         // log(err) GP: 10.0,
                                         // SL(wo kernel):-5.0
                                         // SL(with kernel): 0.0

  const float kGoodFeatureErrVal_ = 0.0;  // log(err)

  // Set to true if you would like to saturate epipolar/reprojection error
  // values to binary values of either kBadFeatureErrVal_ or
  // kGoodFeatureErrVal_
  const bool kMakeErrValuesBinary_ = false;

  // Either the epipolar error threshold or the reprojection error threshold
  // are set to kBadFeatureErrThresh_ given the active error_type_
  float kBadFeatureErrThresh_;

  // The threshold that error values are clamped at for color visualization of
  // the error value. This is used both for generating the heatmap and for
  // color coding error value of the keypoints on the image
  float kErrHeatmapMaxVal_;

  // The minimum and maximum percentage of bad matches in a frame in order
  // for it to be considered useful for training the introspection model
  const float kMinBadMatchPercent_ = 5.0;   // 15, (for 1200 feature extraction)
                                            // 25 (for 2000 feature extraction)
  const float kMaxBadMatchPercent_ = 60.0;  // 75

  // The minimum number of matched features in the frame in order for it to
  // be used in the trainintg dataset.
  const size_t kMinMatchesInFrame_ = 10;

  // Only use matched keypoints whose track length is longer than a threshold
  // NOTE: Currently it is only implemented for
  //       CompareAgainstRefKeyFrameEpipolarNormalized training mode.
  const bool kFilterBasedOnTrackLength_ = false;
  const int kMinTrackLength_ = 20;  // in number of frames

  // Discard keypoints whose epipolar error sensitivity is very low from the
  // training data. It is determined based on the normalization factor that is
  // calculated for reach keypoint. The result being that points that are very
  // close to the epipole will not be used.
  const bool kFilterBasedOnEpipolarErrInformation_ = false;
  const double kMinEpipolarErrSensitivity_ = 1;  // 5,  10

  // If set to true, analytically derived Jacobians will be used to propagate
  // the uncertainty in the reference camera posed as opposed to using the
  // sigma points approach
  const bool kUseAnalyticalUncertaintyPropagation_ = true;

  // Image quality heatmap parameters
  const float kBinSizeX_ = 40.0;  // 200, *100, 50, +40
  const float kBinSizeY_ = 40.0;
  const float kBinStride_ = 20.0;  // +20
  const int kMaxBinFreq_ = 1000;   // 1000, 10

  // If set to true, current frame will not be used for training. This is a
  // means for this to be enforced from an external source such as the
  // tracking accuracy when using unsupervised learning
  Reliability frame_reliability_ = Unknown;

  const bool kDebug_ = true;
  bool camera_calib_loaded_ = false;
  DescriptorType descriptor_type_ = kORB;
  Dataset dataset_ = kKITTI;
  TrainingMode training_mode_ = kCompareAgainstRefKeyFrameEpipolarNormalized;
  VisualizationMode visualization_mode_ = kVisualizeAll;

  // Error type is automatically set given the training mode.
  ErrorType error_type_;

  cv::Mat img_prev_, img_curr_;

  cv::Mat img_matching_annotation_, img_feature_qual_annotation_;

  // Visualizes the normalization factor for each of the extracted image
  // features.
  cv::Mat img_err_normalization_factor_;
  cv::Mat img_bad_matching_annotation_;
  cv::Mat img_reproj_err_vec_;
  cv::Mat img_epipolar_err_vec_;

  // Visualizes the normalization factor value for each extracted image feature
  // over the image
  cv::Mat img_epipolar_err_norm_factor_;
  cv::Mat camera_mat_, dist_coeffs_cv_;
  cv::Mat unrect_map1_left_, unrect_map2_left_;
  bool rectification_map_available_ = false;

  Eigen::Matrix<double, 3, 4, Eigen::DontAlign> projection_mat_cam0_;
  Eigen::Matrix<double, 3, 4, Eigen::DontAlign> projection_mat_cam1_;
  std::vector<uchar> feature_pts_status_;
  cv::Mat descriptors1_, descriptors2_;

  // Has higher values at regions with extracted features that have high error
  // It is the reprojection/epipolar error (err_vals_select_) heatmap after
  // being clamped at a maximum value and then scaled
  cv::Mat bad_region_heatmap_;

  // A binary mask that is non zero at regions where the uncertainty of the
  // calculating heatmap is less than a threshold
  cv::Mat bad_region_heatmap_mask_;

  // Percentage of feature matches that have been flagged as bad
  float bad_matches_percent_ = 0;

  // All keypoints in the current frame (frame2) that are matched
  // with map points and are used for training the introspection model
  // It is the same size as err_vals_select_ and selection is made based on
  // training_mode_
  std::vector<cv::KeyPoint> keypts2_select_;

  // Life time of all keypoints in keypts2_select_
  // (currentFrameID - FirstFrameID)
  std::vector<long int> keypts2_select_life_;

  // Offset of all keypoints in keypts2_select_ from their reference frame
  // (currentFrameID - referenceFrameID)
  std::vector<long int> keypts2_select_ref_frame_offset_;

  // All calculated reprojection/epipolar error values that are used for
  // training the introspection model. It is the same length as keypts2_select_
  // This is the reprojection/epipolar error for a subset of map points that are
  // matched with keypoints in current frame. The subset changes based on
  // the current training_mode_
  std::vector<float> err_vals_select_;

  // The interpolated epipolar error over all the image
  std::vector<double> err_vals_interp_;

  // Reprojection of all corresponding map_points with keypts2_select_ in
  // the current frame(frame2). The ground truth pose of current frame is used
  // for projecting the 3d points into the image plane. The exact method
  // depends on the training_mode_
  std::vector<cv::Point2f> reproj2_select_;

  // The GROUND TRUTH reprojection of all corresponding map_points with
  // keypts2_select_ in the current frame(frame2). The ground truth 3D pose
  // of the map point is calculated using the registered depth of the
  // corresponding keypoint in the reference frame (or previous frame given
  // the training mode). It is then projected to the image plane of current
  // frame using the ground truth camera pose.
  std::vector<cv::Point2f> reproj2_select_gt_;

  // Vector of unit vectors representing direction of epipolar lines
  // corresponding to each of keypts2_select_
  std::vector<Eigen::Vector2f> epipolar_lines_dir_;

  // Vector of points that are the projections of each of keypts2_select_ on
  // their corresponding epipolar line
  std::vector<Eigen::Vector2f> epipolar_projections_;

  // Vector of sigma point errors corresponding to each of keypts2_select_
  // features. These are vectors from the extracted keyoints to projections on
  // the epipolar line corresponding to that specific sigma point.
  // The same size as keypt2_select_
  std::vector<std::vector<Eigen::Vector2f>> sigma_pts_err_;

  // The normalization factor calculated for each of the error values.
  // It is the same size as err_vals_select_. The raw error values equal
  // err_vals_select_ .* err_norm_factor
  std::vector<double> err_norm_factor_;

  // The epipolar error vectors (from keypts2 to the projection on the epipolar
  // line) for all the particles of all keypoints. This is of size N * M, where
  // N = keypts2_select_.size() and M = kParticleSize_
  // std::vector<std::vector<Eigen::Vector2f>> epi_err_vec_particles_;

  // Mean of particles of epipolar error vectors. This is of size N * 1, where
  // N is keypts2_select_.size()
  // std::vector<Eigen::Vector2f> epi_err_vec_particles_mean_;

  // Covariance of particles of epipolar error vectors.
  // This is of size N * 2 * 2, where N is keypts2_select_.size()
  std::vector<Eigen::Matrix2f> epi_err_vec_particles_cov_;

  // Keypoints in the current frame (frame2) and previous frame that are
  // matched with each other. keypts2_matched_w_prev_ is a subset of
  // keypts2_select_ that excludes keyopints in the current frame that are
  // matched with map points that do not have a correspondence with
  // the immediate previous frame
  // In the training mode kMatchesWithRefKeyFrame, keypts2_matched_w_prev_
  // includes keypoints in the current frame that are matched with keypoints
  // in the reference key frame instead of the immediate previous frame.
  std::vector<cv::KeyPoint> keypts1_matched_w_curr_, keypts2_matched_w_prev_;
  std::vector<cv::DMatch> matches1to2_;

  // The vector of reprojection/epipolar errors that is only used for
  // visualization. It only includes those keypoints for which a match has been
  // found on the previous image(for the kMatchesWithPrevFrame training mode) or
  // the reference keyframe (for the kMatchesWithRefKeyFrame train. mode)
  // It is the same length as keypts2_matched_w_prev_
  std::vector<float> err_vals_visualization_;

  // Subsets of keypts1_matched_w_curr_ and keypts2_matched_w_prev_ that have
  // high reprojection/epipolar error. Used for visualization purposes.
  std::vector<cv::KeyPoint> bad_keypts1_matched_w_curr_;
  std::vector<cv::KeyPoint> bad_keypts2_matched_w_prev_;
  std::vector<cv::DMatch> bad_matches1to2_;

  bool rel_cam_pose_uncertainty_available_ = false;
  const std::unordered_map<std::string, int>* rel_cam_poses_unc_map_;
  const std::vector<Eigen::Vector2f>* rel_cam_poses_unc_;

  void EvaluateAgainstPrevFrame(ORB_SLAM2::Frame& prev_frame,
                                ORB_SLAM2::Frame& curr_frame);

  void EvaluateAgainstPrevFrameAndLastKeyFrame(ORB_SLAM2::Frame& prev_frame,
                                               ORB_SLAM2::Frame& curr_frame);

  void EvaluateAgainstPrevFrameAndRefKeyFrame(ORB_SLAM2::Frame& prev_frame,
                                              ORB_SLAM2::Frame& curr_frame);

  void EvaluateAgainstLastKeyFrame(ORB_SLAM2::Frame& prev_frame,
                                   ORB_SLAM2::Frame& curr_frame);

  void EvaluateAgainstRefKeyFrame(ORB_SLAM2::Frame& prev_frame,
                                  ORB_SLAM2::Frame& curr_frame);

  // Calculates the epipolar error rather than the reprojection error for
  // keypoints that exist in the current frame
  void EvaluateAgainstPrevFrameEpipolar(ORB_SLAM2::Frame& prev_frame,
                                        ORB_SLAM2::Frame& curr_frame);

  // Calculates the epipolar error rather than the reprojection error for
  // keypoints that exist in the current frame
  void EvaluateAgainstRefKeyFrameEpipolar(ORB_SLAM2::Frame& prev_frame,
                                          ORB_SLAM2::Frame& curr_frame);

  // Calculates the epipolar error rather than the reprojection error for
  // keypoints that exist in the current frame. It also uses sigma points
  // for calculating the covariance of error and uses that for
  // normalization. Only keypoints that are matched with a point in the
  // previous frame are used.
  void EvaluateAgainstPrevFrameEpipolarNormalized(ORB_SLAM2::Frame& prev_frame,
                                                  ORB_SLAM2::Frame& curr_frame);

  // Calculates the epipolar error rather than the reprojection error for
  // keypoints that exist in the current frame. It also uses sigma points
  // for calculating the covariance of error and uses that for
  // normalization.
  void EvaluateAgainstRefKeyFrameEpipolarNormalized(
      ORB_SLAM2::Frame& prev_frame, ORB_SLAM2::Frame& curr_frame);

  // Calculates the epipolar error (both the scalar cos(err_angle) and the
  // error vector in pixels) for a pair of keypoints given the
  // ground truth relative pose of the camera.
  Eigen::Vector2f CalculateEpipolarErrorVec(
      const Eigen::Matrix3d& R_prev_to_curr,
      const Eigen::Vector3d& trans_prev_to_curr,
      const ORB_SLAM2::Frame& curr_frame,
      const cv::KeyPoint& keypoint1,
      const cv::KeyPoint& keypoint2,
      float* epipolar_err_scalar,
      Eigen::Vector2f* epipolar_line_dir,
      Eigen::Vector2f* proj_on_epipolar_line);

  // Calculates the epipolar error for a pair of keypoints given their
  // corresponding frame (which includes the ground truth pose of camera)
  double CalculateEpipolarError(const ORB_SLAM2::KeyFrame& ref_keyframe,
                                const ORB_SLAM2::Frame& curr_frame,
                                const cv::KeyPoint& keypoint1,
                                const cv::KeyPoint& keypoint2,
                                Eigen::Vector2f* epipolar_line_dir,
                                Eigen::Vector2f* proj_on_epipolar_line);

  // Calculates the epipolar error given a pair of keypoints and the
  // ground truth transformation from one frame to the other
  double CalculateEpipolarError(const cv::Mat& tf_prev_to_curr,
                                const ORB_SLAM2::Frame& curr_frame,
                                const cv::KeyPoint& keypoint1,
                                const cv::KeyPoint& keypoint2,
                                Eigen::Vector2f* epipolar_line_dir,
                                Eigen::Vector2f* proj_on_epipolar_line);

  // Calculates the 3D coordinates of the points on the unit shpere
  // given the pixel coordinates of the point on the image
  void CalculateNormalized3DPoint(const cv::Point2f& pixel_coord,
                                  const int cam_id,
                                  Eigen::Vector3d* normalized_3d_coord);

  // Calculates the reprojection error given the keypoint in the current frame
  // and the corresponding map point. The reprojection error is calculated
  // taking into account the ground truth pose of current frame with respect
  // to a reference frame
  double CalculateReprojectionError(const ORB_SLAM2::Frame& ref_frame,
                                    const ORB_SLAM2::Frame& curr_frame,
                                    const cv::Mat& tf_prev_to_curr,
                                    ORB_SLAM2::MapPoint& map_point,
                                    const cv::KeyPoint& keypoint,
                                    cv::Point2f* reproj_pt);

  double CalculateReprojectionError(const ORB_SLAM2::Frame& ref_frame,
                                    const ORB_SLAM2::Frame& curr_frame,
                                    ORB_SLAM2::MapPoint& map_point,
                                    const cv::KeyPoint& keypoint,
                                    cv::Point2f* reproj_pt);

  double CalculateReprojectionError(ORB_SLAM2::KeyFrame& ref_frame,
                                    const ORB_SLAM2::Frame& curr_frame,
                                    ORB_SLAM2::MapPoint& map_point,
                                    const cv::KeyPoint& keypoint,
                                    cv::Point2f* reproj_pt);

  cv::Mat GenerateErrHeatmap(unsigned int rows,
                             unsigned int cols,
                             const std::vector<double> err_vals);

  cv::Mat GenerateErrHeatmap(unsigned int rows,
                             unsigned int cols,
                             const std::vector<double> err_vals,
                             double err_max_clamp,
                             double err_min_clamp);

  // Distributes a set of input points in the corresponding bins. Returns
  // both a matrix of frequencies and the sum of values in each bin.
  // The provided freq and bin_val nested vectors should have already been
  // resized to the number of bins.
  void Hist2D(const std::vector<double>& x_in,
              const std::vector<double>& y_in,
              const std::vector<float>& values,
              double bin_size_x,
              double bin_size_y,
              double stride_x,
              double stride_y,
              int bin_num_x,
              int bin_num_y,
              Eigen::ArrayXXd* freq,
              Eigen::ArrayXXd* bin_val);

  Eigen::ArrayXXd ClampArray(const Eigen::ArrayXXd& input_array, int max_value);

  cv::Mat OverlayHeatmapOnImage(const cv::Mat& heatmap);

  // Projects a 3D point in the camera reference frame to the image plane of
  // the camera
  cv::Mat ProjectToCam(const ORB_SLAM2::Frame& cam_frame,
                       const cv::Mat& point_3d_in_cam_ref);

  Eigen::Vector2f ProjectToCam(const ORB_SLAM2::Frame& cam_frame,
                               const Eigen::Vector3f& point_3d_in_cam_ref);

  // Unrectifies the input image if a corresponding map has been generated
  // given the camera calibration parameters. It applies the rectification map
  // for the left camera
  cv::Mat UnrectifyImage(const cv::Mat& input_img);
};

}  // namespace feature_evaluation

#endif  // IVSLAM_FEATURE_EVALUATOR
