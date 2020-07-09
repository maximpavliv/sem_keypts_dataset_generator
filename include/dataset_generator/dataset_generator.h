#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <random_numbers/random_numbers.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <eigen_conversions/eigen_msg.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"

using namespace Eigen;
using namespace std;

namespace image_processor
{
    class ImageProcessor
    {
        public:
            ImageProcessor(ros::NodeHandle& nh, ros::NodeHandle& nh_priv);
            ~ImageProcessor();
        private:
            // ros node handle
            ros::NodeHandle nh_;
            ros::NodeHandle nh_priv_;
           
            std::vector<string> odometry_objects; // First: camera; second: main drone
            
            // image subscriber
            image_transport::ImageTransport it_;
            image_transport::Subscriber image_sub_;

            image_transport::Publisher image_full_pub_;
            image_transport::Publisher image_cropped_pub_;

            bool first_frame_ready_ = false;
            bool to_record = false;
            string root_;
            int record_idx_ = 0;
            fstream myfile_;

            Eigen::Matrix3d R_cam_world_;

            Eigen::Vector3d T_offset_;
            Eigen::Matrix3d R_offset_;
            ros::Duration time_offset_tune;

            std::vector<ros::Subscriber> odometry_subs_;

            
            // parameters for images
            int im_height_, im_width_;
            double quad_height_, quad_width_;
            cv::Vec4d intrinsics_ = cv::Vec4d(565.2027524412007, 565.6613573390968, 492.48969368881677, 271.0003077726479);
            Eigen::Matrix3d K;
            cv::Vec4d dist_coeff_ = cv::Vec4d(0.3228426067274788, 0.6782271574023381, -2.7508173110174234, 3.647246884438062);

            // image callback
            void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);
            // utility funcs for loading params
            void loadParam();
            Eigen::Matrix<double, 3,3> yprToRot(const Eigen::Matrix<double,3,1>& ypr);
             
            std::vector<ros::Duration> ts_offsets_sync_;

            std::vector<std::vector<nav_msgs::OdometryConstPtr>> odometry_buffers;

            void odometry_callback(const nav_msgs::OdometryConstPtr& odometry_msg);

            void processImage(const cv_bridge::CvImageConstPtr& img_msg);

    };
}


#endif