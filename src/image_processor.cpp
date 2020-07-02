#include "image_processor/image_processor.h"

namespace image_processor
{
	ImageProcessor::ImageProcessor(ros::NodeHandle& nh, ros::NodeHandle& nh_priv):
		nh_(nh),
		nh_priv_(nh_priv),
		it_(nh_priv),
		gt_sync_(ApproximateTimePolicy(10), glasses_odom_sub_, drone_odom_sub_)
	{
		R_cam_world_ << 0.0, -1.0, 0.0,
						0.0, 0.0, -1.0,
						1.0, 0.0, 0.0;
		loadParam();

		glasses_odom_sub_.subscribe(nh_priv_, "glasses_odom_topic", 10);
		drone_odom_sub_.subscribe(nh_priv_, "drone_odom_topic", 10);
		image_sub_ = it_.subscribe("image", 1, &ImageProcessor::imageCallback, this);
		gt_sync_.registerCallback(boost::bind(&ImageProcessor::poseGTCallback, this, _1, _2));
		image_full_pub_ = it_.advertise("image_full", 1);
		image_cropped_pub_ = it_.advertise("image_cropped", 1);

		if(to_record)
		{
			std::string string0 = root_ + "annotation_bag_6.txt";
			myfile_.open(string0, ios::out);

			cout << string0 << endl;
			assert(myfile_.is_open());
		}

	}

	void ImageProcessor::loadParam()
	{
		nh_priv_.param<int>("image_height", im_height_, 1080/2);
		nh_priv_.param<int>("image_width", im_width_, 1920/2);
		nh_priv_.param<string>("fixed_frame_id", fixed_frame_id_, "map");
		nh_priv_.param<bool>("is_record", to_record, false);
		nh_priv_.param<int>("record_skip_frames", record_skip_frames_, 5);
		nh_priv_.param<string>("root", root_, "/media/mrsl/liangzhe-T5/drone_dataset/images/");

		double offset_x, offset_y, offset_z;
		nh_priv_.param<double>("offset_x", offset_x, 0.0);
		nh_priv_.param<double>("offset_y", offset_y, 0.0);
		nh_priv_.param<double>("offset_z", offset_z, 0.0);

		double offset_yaw, offset_pitch, offset_roll;
		nh_priv_.param<double>("offset_roll", offset_roll, 0.0);
		nh_priv_.param<double>("offset_pitch", offset_pitch, 0.0);
		nh_priv_.param<double>("offset_yaw", offset_yaw, 0.0);
		Eigen::Vector3d ypr(offset_yaw, offset_pitch, offset_roll);

		nh_priv_.param<double>("boundbox_scale", bbox_scale_, 1.0);

		double delay;
		// delay in ms
		nh_priv_.param<double>("delay", delay, 200);
		// assume video frame rate to be 25Hz
		skip_frames_ = delay/1;

		nh_priv_.param<double>("quad_width", quad_width_, 0.2);
		nh_priv_.param<double>("quad_height", quad_height_, 0.08);
		T_offset_ << offset_x, offset_y, offset_z;
		R_offset_ = yprToRot(ypr);
		R_cam_world_ = R_cam_world_ * R_offset_;
	}

	void ImageProcessor::poseGTCallback(const nav_msgs::OdometryConstPtr& glasses_odom,
										const nav_msgs::OdometryConstPtr& drone_odom)
	{
		GTPair gt_odoms(*glasses_odom, *drone_odom);
		gt_buffer_.push_back(gt_odoms);
	}

	void ImageProcessor::imageCallback(const sensor_msgs::ImageConstPtr& img_msg)
	{
		try
		{
			++ frame_counter_;
			image_ptr_ = cv_bridge::toCvShare(img_msg, "bgr8");
			processImage(image_ptr_, gt_buffer_);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("Could not convert from '%s' to 'mono8'.", img_msg->encoding.c_str());
		}

	}


	void ImageProcessor::processImage(const cv_bridge::CvImageConstPtr& img_msg,
									  std::vector<GTPair>& gt_buffer)
	{
		static Eigen::Matrix3d K;
		K << intrinsics_[0], 0, intrinsics_[2],
			 0, intrinsics_[1], intrinsics_[3],
			 0, 0, 1;
		static double focal_length = (intrinsics_[0] + intrinsics_[1])/2;
		static char filename_buffer[100];
		static char annotation_buffer[100];
		if(!first_frame_ready_)
		{
			if(frame_counter_ < skip_frames_)
			{
				ROS_WARN("SKIP ONE FRAME");
				return;
			}
			if(gt_buffer.empty())
				return;
			ts_offset_ = img_msg->header.stamp - gt_buffer.front().first.header.stamp;
			first_frame_ready_ = true;
			frame_counter_ = 0;
			return;
		}
		ros::Time ts_now = img_msg->header.stamp - ts_offset_;
		auto iter = gt_buffer.begin();
		while(iter != gt_buffer.end() - 1)
		{
			if(iter->first.header.stamp < ts_now)
				++iter;
			else
				break;
		}
		if (iter != gt_buffer.begin()) iter--;

		GTPair gt_now = *iter;
		gt_buffer.erase(gt_buffer.begin(), iter);
		
		// Transform the ground truth.
		Eigen::Quaterniond orientation;
		Eigen::Vector3d translation;
		
		// convert glasses odom into original frame 
		tf::pointMsgToEigen(gt_now.first.pose.pose.position, translation); 
		tf::quaternionMsgToEigen(gt_now.first.pose.pose.orientation, orientation); 
		Eigen::Isometry3d H_ref_glasses;
		H_ref_glasses.linear() = orientation.toRotationMatrix();
		H_ref_glasses.translation() = translation;

		// convert drone odom into original frame
		tf::pointMsgToEigen(gt_now.second.pose.pose.position, translation);
		tf::quaternionMsgToEigen(gt_now.second.pose.pose.orientation, orientation);
		Eigen::Isometry3d H_ref_drone;
		H_ref_drone.linear() = orientation.toRotationMatrix();
		H_ref_drone.translation() = translation + T_offset_;

	
		Eigen::Isometry3d H_glasses_drone = H_ref_glasses.inverse() * H_ref_drone;
		
		Eigen::Vector3d T_glasses_drone = H_glasses_drone.translation();

		// convert translation vector into cam frame
		T_glasses_drone = R_cam_world_*T_glasses_drone;

		Eigen::Vector3d projective_T = K*T_glasses_drone;

		cv::Point2d point(projective_T(0)/projective_T(2),
						  projective_T(1)/projective_T(2));

		cv::Mat image;
		cv::resize(img_msg->image, image, cv::Size(im_width_, im_height_), 0, 0);
		// draw the box


		//Computing keypoints of 3d box around the drone
		Eigen::Matrix3d rotationM = orientation.toRotationMatrix();
		std::vector<Eigen::Vector3d> keypoints_drone_frame = {Eigen::Vector3d(quad_width_/2*bbox_scale_,quad_width_/2*bbox_scale_,quad_height_/2*bbox_scale_),
														   Eigen::Vector3d(quad_width_/2*bbox_scale_,quad_width_/2*bbox_scale_,-quad_height_/2*bbox_scale_),
														   Eigen::Vector3d(quad_width_/2*bbox_scale_,-quad_width_/2*bbox_scale_,quad_height_/2*bbox_scale_), 
														   Eigen::Vector3d(quad_width_/2*bbox_scale_,-quad_width_/2*bbox_scale_,-quad_height_/2*bbox_scale_), 
														   Eigen::Vector3d(-quad_width_/2*bbox_scale_,quad_width_/2*bbox_scale_,quad_height_/2*bbox_scale_), 
														   Eigen::Vector3d(-quad_width_/2*bbox_scale_,quad_width_/2*bbox_scale_,-quad_height_/2*bbox_scale_), 
														   Eigen::Vector3d(-quad_width_/2*bbox_scale_,-quad_width_/2*bbox_scale_,quad_height_/2*bbox_scale_), 
														   Eigen::Vector3d(-quad_width_/2*bbox_scale_,-quad_width_/2*bbox_scale_,-quad_height_/2*bbox_scale_)};
		
	//    std::vector<Eigen::Vector3d> keypoints_global_frame;
		std::vector<cv::Point2d> keypoints_camera_frame;
		
		for(int i=0;i<8;i++)
		{
			Eigen::Vector3d T_glasses_drone_edge = H_glasses_drone*keypoints_drone_frame[i];
			
			// convert translation vector into cam frame
			T_glasses_drone_edge = R_cam_world_*T_glasses_drone_edge;
			Eigen::Vector3d projective_T_edge = K*T_glasses_drone_edge;
			cv::Point2d point_edge(projective_T_edge(0)/projective_T_edge(2),
							  projective_T_edge(1)/projective_T_edge(2));
			keypoints_camera_frame.push_back(point_edge);
		}

		double bbox_xmin = keypoints_camera_frame[0].x;
		double bbox_ymin = keypoints_camera_frame[0].y;
		double bbox_xmax = keypoints_camera_frame[0].x;
		double bbox_ymax = keypoints_camera_frame[0].y;
		for (int i = 1; i < 8; i++) {
			bbox_xmin = std::min(bbox_xmin, keypoints_camera_frame[i].x);
			bbox_xmax = std::max(bbox_xmax, keypoints_camera_frame[i].x);
			bbox_ymin = std::min(bbox_ymin, keypoints_camera_frame[i].y);
			bbox_ymax = std::max(bbox_ymax, keypoints_camera_frame[i].y);
		}
		double margin_bbox = 20;
		bbox_xmin-=margin_bbox; bbox_ymin-= margin_bbox; bbox_xmax+=margin_bbox; bbox_ymax+=margin_bbox;  

		cv::Mat image_full;
		image.copyTo(image_full);


		//Draw center of keypoints volume
		cv::circle(image_full, point, 1 , cv::Scalar(0, 255, 255), 6);

		//Draw keypoints volume
		cv::line(image_full, keypoints_camera_frame[0], keypoints_camera_frame[4], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[1], keypoints_camera_frame[5], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[2], keypoints_camera_frame[6], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[3], keypoints_camera_frame[7], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[0], keypoints_camera_frame[2], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[1], keypoints_camera_frame[3], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[4], keypoints_camera_frame[6], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[5], keypoints_camera_frame[7], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[0], keypoints_camera_frame[1], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[2], keypoints_camera_frame[3], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[4], keypoints_camera_frame[5], cv::Scalar(100, 100, 0), 0.5);
		cv::line(image_full, keypoints_camera_frame[6], keypoints_camera_frame[7], cv::Scalar(100, 100, 0), 0.5);

		//Draw bbox
		cv::rectangle(image_full, cv::Point2d(bbox_xmin, bbox_ymin), cv::Point2d(bbox_xmax, bbox_ymax), cv::Scalar(255,0,0), 0.5);
		
		//draw keypoints
		for (int i = 0; i < 8; i++) {
			if(i != 5 && i != 7)
				cv::circle(image_full, keypoints_camera_frame[i], 1 , cv::Scalar(255, 255, 0), 6); 
		}
		cv::circle(image_full, keypoints_camera_frame[5], 1 , cv::Scalar(255, 0, 255), 6); 
		cv::circle(image_full, keypoints_camera_frame[7], 1 , cv::Scalar(255, 0, 255), 6); 


		//Publish bbox image
		cv_bridge::CvImage image_full_bridge;
		image_full_bridge.header.stamp = ts_now;
		image_full_bridge.image = image_full;
		image_full_bridge.encoding = sensor_msgs::image_encodings::BGR8;
		sensor_msgs::ImagePtr full_msg = image_full_bridge.toImageMsg();
		image_full_pub_.publish(full_msg);

		//Cropped image
		int square_bbox_xmin, square_bbox_ymin, square_bbox_xmax, square_bbox_ymax;
		int square_bbox_width;
		if((bbox_xmax-bbox_xmin) > (bbox_ymax-bbox_ymin))
		{
			square_bbox_width = bbox_xmax-bbox_xmin;
			square_bbox_xmin = bbox_xmin;
			square_bbox_xmax = bbox_xmax;
			square_bbox_ymin = (bbox_ymax+bbox_ymin)/2 - square_bbox_width/2;
			square_bbox_ymax = (bbox_ymax+bbox_ymin)/2 + square_bbox_width/2;
		}
		else
		{
			square_bbox_width = bbox_ymax-bbox_ymin;
			square_bbox_ymin = bbox_ymin;
			square_bbox_ymax = bbox_ymax;
			square_bbox_xmin = (bbox_xmax+bbox_xmin)/2 - square_bbox_width/2;
			square_bbox_xmax = (bbox_xmax+bbox_xmin)/2 + square_bbox_width/2;
		}

		if(square_bbox_xmin>=0 && square_bbox_ymin >= 0 && square_bbox_xmax < image.cols && square_bbox_ymax < image.rows)
		{
			cv::Mat image_cropped;
			image(cv::Rect(square_bbox_xmin,square_bbox_ymin,square_bbox_width,square_bbox_width)).copyTo(image_cropped);

			cv::Point2d square_coordinates_offset(square_bbox_xmin, square_bbox_ymin);

			std::vector<cv::Point2d> keypoints_cropped;
			for(int i=0; i<8; i++)
			{
				cv::Point2d kpt_cropped;
				kpt_cropped = keypoints_camera_frame[i] - square_coordinates_offset;
				keypoints_cropped.push_back(kpt_cropped);
			}	

					//Draw keypoints volume
			cv::line(image_cropped, keypoints_cropped[0], keypoints_cropped[4], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[1], keypoints_cropped[5], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[2], keypoints_cropped[6], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[3], keypoints_cropped[7], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[0], keypoints_cropped[2], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[1], keypoints_cropped[3], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[4], keypoints_cropped[6], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[5], keypoints_cropped[7], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[0], keypoints_cropped[1], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[2], keypoints_cropped[3], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[4], keypoints_cropped[5], cv::Scalar(100, 100, 0), 0.5);
			cv::line(image_cropped, keypoints_cropped[6], keypoints_cropped[7], cv::Scalar(100, 100, 0), 0.5);

			for (int i = 0; i < 8; i++) {
				if(i != 5 && i != 7)
					cv::circle(image_cropped, keypoints_cropped[i], 1 , cv::Scalar(255, 255, 0), 1); 
			}
			cv::circle(image_cropped, keypoints_cropped[5], 1 , cv::Scalar(255, 0, 255), 1); 
			cv::circle(image_cropped, keypoints_cropped[7], 1 , cv::Scalar(255, 0, 255), 1); 


			//Publish cropped image
			cv_bridge::CvImage image_cropped_bridge;
			image_cropped_bridge.header.stamp = ts_now;
			image_cropped_bridge.image = image_cropped;
			image_cropped_bridge.encoding = sensor_msgs::image_encodings::BGR8;
			sensor_msgs::ImagePtr cropped_msg = image_cropped_bridge.toImageMsg();
			image_cropped_pub_.publish(cropped_msg);
		}

		if(to_record)
		{
			if((frame_counter_%skip_frames_==0) && bbox_xmin > 0 && bbox_ymin > 0 && bbox_xmax < im_width_ && bbox_ymax < im_height_ )
			{
				/*
				std::vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(3);
				snprintf(filename_buffer, sizeof(filename_buffer), (root_ + "drone_raw_6/auto_dataset_bag_6_%06d.png").c_str(), record_idx_);
				std::string file_path = filename_buffer;
				cv::imwrite(file_path, image, compression_params);
				snprintf(filename_buffer, sizeof(filename_buffer), (root_ + "drone_bbox_6/auto_dataset_bag_6_%06d.png").c_str(), record_idx_);
				file_path = filename_buffer;
				cv::imwrite(file_path, image_bbox, compression_params);


				snprintf(annotation_buffer, sizeof(annotation_buffer), "auto_dataset_bag_6_%06d.png,0 %.10f %.10f %.10f %.10f", record_idx_, (xmax+xmin)/im_width_/2, (ymax+ymin)/2/im_height_, (xmax-xmin)/im_width_, (ymax-ymin)/im_height_);
				std::string annotation_string = annotation_buffer;
				myfile_ << annotation_string << "\n";

				std::cout << "Writing image to:" << file_path << std::endl;
				++record_idx_;
				frame_counter_ = 0;
				*/
			}
		}

		return;
	}

	Eigen::Matrix<double, 3,3> ImageProcessor::yprToRot(const Eigen::Matrix<double,3,1>& ypr)
	{
		double c, s;
		Eigen::Matrix<double,3,3> Rz;
		Rz.setZero();
		double y = ypr(0,0);
		c = cos(y);
		s = sin(y);
		Rz(0,0) =  c;
		Rz(1,0) =  s;
		Rz(0,1) = -s;
		Rz(1,1) =  c;
		Rz(2,2) =  1;

		Eigen::Matrix<double,3,3> Ry;
		Ry.setZero();
		double p = ypr(1,0);
		c = cos(p);
		s = sin(p);
		Ry(0,0) =  c;
		Ry(2,0) = -s;
		Ry(0,2) =  s;
		Ry(2,2) =  c;
		Ry(1,1) =  1;

		Eigen::Matrix<double, 3,3> Rx;
		Rx.setZero();
		double r = ypr(2,0);
		c = cos(r);
		s = sin(r);
		Rx(1,1) =  c;
		Rx(2,1) =  s;
		Rx(1,2) = -s;
		Rx(2,2) =  c;
		Rx(0,0) =  1;

		Eigen::Matrix<double, 3,3> R = Rz*Ry*Rx;
		return R;
	}

	ImageProcessor::~ImageProcessor()
	{
		std::cout << record_idx_ << " frames have been saved to disk!" << std::endl;
		myfile_.close();
	}

}
