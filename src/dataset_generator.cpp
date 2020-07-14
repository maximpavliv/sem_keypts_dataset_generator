#include "dataset_generator/dataset_generator.h"

namespace image_processor
{
	ImageProcessor::ImageProcessor(ros::NodeHandle& nh, ros::NodeHandle& nh_priv):
		nh_(nh),
		nh_priv_(nh_priv),
		it_(nh_priv)
	{
		R_cam_world_ << 0.0, -1.0, 0.0,
						0.0, 0.0, -1.0,
						1.0, 0.0, 0.0;						
		loadParam();
		K << intrinsics_[0], 0, intrinsics_[2],
        	 0, intrinsics_[1], intrinsics_[3],
             0, 0, 1;

		image_sub_ = it_.subscribe("image", 1, &ImageProcessor::imageCallback, this);

		image_full_pub_ = it_.advertise("image_full", 1);
		image_cropped_pub_ = it_.advertise("image_cropped", 1);

		for(int i=0; i<(int)odometry_objects.size(); i++)
		{
			ros::Subscriber new_odom_sub = nh_priv_.subscribe("/vicon/" + odometry_objects[i] + "/odom", 1, &ImageProcessor::odometry_callback, this); // /vicon/TobiiGlasses/odom
			odometry_subs_.push_back(new_odom_sub);
			std::vector<nav_msgs::OdometryConstPtr> odometry_buffer;
			odometry_buffers.push_back(odometry_buffer);
		}

		if(to_record)
		{
			std::string absolut_filename = root_ + filename_;
			myfile_.open(absolut_filename, ios::out);
			assert(myfile_.is_open());
		}

	}

	void ImageProcessor::loadParam()
	{
		nh_priv_.getParam("odometry_objects", odometry_objects);
		nh_priv_.param<int>("image_height", im_height_, 1080/2);
		nh_priv_.param<int>("image_width", im_width_, 1920/2);
		nh_priv_.param<bool>("to_record", to_record, false);
		nh_priv_.param<string>("root", root_, "/home/max/sem_keypts_root/");
		nh_priv_.param<string>("filename", filename_, "annotations.txt");

		double offset_x, offset_y, offset_z;
		nh_priv_.param<double>("offset_x", offset_x, 0.0);
		nh_priv_.param<double>("offset_y", offset_y, 0.0);
		nh_priv_.param<double>("offset_z", offset_z, 0.0);

		double offset_yaw, offset_pitch, offset_roll;
		nh_priv_.param<double>("offset_roll", offset_roll, 0.0);
		nh_priv_.param<double>("offset_pitch", offset_pitch, 0.0);
		nh_priv_.param<double>("offset_yaw", offset_yaw, 0.0);
		Eigen::Vector3d ypr(offset_yaw, offset_pitch, offset_roll);

		double t_offset_tune;
		nh_priv_.param<double>("time_offset", t_offset_tune, 0.0);
		time_offset_tune = ros::Duration(t_offset_tune);

		nh_priv_.param<double>("quad_width", quad_width_, 0.2);
		nh_priv_.param<double>("quad_height", quad_height_, 0.08);

		T_offset_ << offset_x, offset_y, offset_z;
		R_offset_ = yprToRot(ypr);
		R_cam_world_ = R_cam_world_ * R_offset_;

		keypoints_drone_frame = {Eigen::Vector3d(quad_width_/2,quad_width_/2,quad_height_/2),
								 Eigen::Vector3d(quad_width_/2,quad_width_/2,-quad_height_/2),
								 Eigen::Vector3d(quad_width_/2,-quad_width_/2,quad_height_/2), 
								 Eigen::Vector3d(quad_width_/2,-quad_width_/2,-quad_height_/2), 
								 Eigen::Vector3d(-quad_width_/2,quad_width_/2,quad_height_/2), 
								 Eigen::Vector3d(-quad_width_/2,quad_width_/2,-quad_height_/2), 
								 Eigen::Vector3d(-quad_width_/2,-quad_width_/2,quad_height_/2), 
								 Eigen::Vector3d(-quad_width_/2,-quad_width_/2,-quad_height_/2)};

	}

	void ImageProcessor::odometry_callback(const nav_msgs::OdometryConstPtr& odometry_msg)
	{
		auto object = odometry_msg->child_frame_id.substr(0,odometry_msg->child_frame_id.size()-10);
		bool found_obj = false;
		for(int i=0; i<(int)odometry_buffers.size(); i++)
		{
			if(object == odometry_objects[i])
			{
				found_obj=true;
				odometry_buffers[i].push_back(odometry_msg);
				break;		
			}
		}
		if(!found_obj)
			ROS_WARN("Odometry message from unknown object");
	}

	void ImageProcessor::imageCallback(const sensor_msgs::ImageConstPtr& img_msg)
	{
		try
		{
			cv_bridge::CvImageConstPtr image_ptr_ = cv_bridge::toCvShare(img_msg, "bgr8");;
			processImage(image_ptr_);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("Could not convert from '%s' to 'mono8'.", img_msg->encoding.c_str());
		}

	}

	void ImageProcessor::processImage(const cv_bridge::CvImageConstPtr& img_msg)
	{
		if(!first_frame_ready_)
		{
			if(someOdomBufferIsEmpty())
				return;

			ts_offset_sync_ = img_msg->header.stamp - odometry_buffers.front().front()->header.stamp; 
			
			first_frame_ready_ = true;
		
			return;
		}

		if(someOdomBufferIsEmpty())
		{
			ROS_ERROR("Some odometry buffer is empty");
			return;
		}

		std::vector<nav_msgs::OdometryConstPtr> current_objects_odoms;

		ros::Time ts_now = img_msg->header.stamp - ts_offset_sync_ + time_offset_tune; 

		for(int i=0;i<(int)odometry_buffers.size();i++)
		{
			int j = 0;
			while(j < (int)odometry_buffers[i].size())//WARNING! WAS .end() -1 ORIGINALLY
			{
				if(odometry_buffers[i][j]->header.stamp < ts_now)
					j++;
				else
					break;
			}
			if (j != 0) j--;

			auto obj_odom_now = odometry_buffers[i][j];
			current_objects_odoms.push_back(obj_odom_now);
			odometry_buffers[i].erase(odometry_buffers[i].begin(), odometry_buffers[i].begin()+j);
		}

		std::vector<Eigen::Isometry3d> objects_H_refs;
		std::vector<Eigen::Quaterniond> objects_orientations;		
		for(int i=0;i<(int)odometry_buffers.size();i++)
		{
			Eigen::Vector3d obj_translation;
			Eigen::Quaterniond obj_orientation;
			objects_orientations.push_back(obj_orientation);
			tf::pointMsgToEigen(current_objects_odoms[i]->pose.pose.position, obj_translation); 
			tf::quaternionMsgToEigen(current_objects_odoms[i]->pose.pose.orientation, obj_orientation); 
			Eigen::Isometry3d obj_H_ref;
			obj_H_ref.linear() = obj_orientation.toRotationMatrix();
			obj_H_ref.translation() = obj_translation;
			objects_H_refs.push_back(obj_H_ref);
		}
		objects_H_refs[0].translation() += T_offset_; //WARNING! ORIGINALLY OFFSET WAS ON DRONE AND NOT ON THE GLASSES!


		cv::Mat image;
		cv::resize(img_msg->image, image, cv::Size(im_width_, im_height_), 0, 0);


		//std::vector<cv::Point2d> main_drone_keypoints_camera_frame;

		cv::Mat image_full;
		image.copyTo(image_full);


		std::vector<cv::Point2d> keypoints_camera_frame_main_drone;
		std::vector<double> bbox_limits_main_drone;
		for(int i=1;i<(int)odometry_buffers.size();i++)
		{
			//Compute transform matrix
			Eigen::Isometry3d H_glasses_drone = objects_H_refs[0].inverse() * objects_H_refs[i];
			
			cv::Point2d central_point = computeCentralPoint(H_glasses_drone);

			//Computing keypoints of 3d box around the drone
			//Eigen::Matrix3d rotationM = objects_orientations[i].toRotationMatrix();
			
			std::vector<cv::Point2d> keypoints_camera_frame = computeKeypointsCameraFrame(H_glasses_drone);
			if (i==1)
				keypoints_camera_frame_main_drone = keypoints_camera_frame;

			//xmin, ymin, xmax, ymax
			std::vector<double> bbox_limits = computeBboxLimits(keypoints_camera_frame);
			if(i==1)
				bbox_limits_main_drone = bbox_limits;
			//double bbox_xmin = bbox_limits[0]; double bbox_ymin = bbox_limits[1]; double bbox_xmax = bbox_limits[2]; double bbox_ymax = bbox_limits[3]; 
			
			drawDroneOnFullImage(image_full, central_point, keypoints_camera_frame, bbox_limits);

			cv::putText(image_full, std::to_string(counter), cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 0, 0), 1);
		}

		publishImage(image_full_pub_, image_full, ts_now);

		std::vector<int> square_bbox_limits = computeSquareBboxLimits(bbox_limits_main_drone);
		int square_bbox_xmin = square_bbox_limits[0]; int square_bbox_ymin = square_bbox_limits[1]; int square_bbox_xmax = square_bbox_limits[2]; int square_bbox_ymax = square_bbox_limits[3]; 

		if(square_bbox_xmin>=0 && square_bbox_ymin >= 0 && square_bbox_xmax < image.cols && square_bbox_ymax < image.rows)
		{
			cv::Mat image_cropped;
			int square_bbox_width = square_bbox_xmax - square_bbox_xmin;
			image(cv::Rect(square_bbox_xmin,square_bbox_ymin,square_bbox_width,square_bbox_width)).copyTo(image_cropped);

			cv::Point2d square_coordinates_offset(square_bbox_xmin, square_bbox_ymin);

			std::vector<cv::Point2d> keypoints_cropped = computeKeypointsCropped(keypoints_camera_frame_main_drone, square_coordinates_offset);

			drawDroneOnCroppedImage(image_cropped, keypoints_cropped);

			cv::putText(image_cropped, std::to_string(counter), cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 0, 0), 1);


			publishImage(image_cropped_pub_, image_cropped, ts_now);
		}


		if(to_record)
		{
			if(square_bbox_xmin > 0 && square_bbox_ymin > 0 && square_bbox_xmax < im_width_ && square_bbox_ymax < im_height_ )
			{
				myfile_ << counter << " ";
				myfile_ << (int)((square_bbox_xmin+square_bbox_xmax)/2) << " " << (int)((square_bbox_ymin+square_bbox_ymax)/2) << " ";
				myfile_ << (float)((square_bbox_xmax-square_bbox_xmin))/200 << " ";
				for (int j=0;j<8;j++)
				{
					myfile_ << keypoints_camera_frame_main_drone[j].x << " " << keypoints_camera_frame_main_drone[j].y << " ";
				}
				myfile_ << endl;
				//TODO print out: counter, center, scale, parts.
				//static char filename_buffer[100];
				//static char annotation_buffer[100]; //CHECK DIFFERENCE BETWEEN BOTH
				
				//std::vector<int> compression_params;
				//compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				//compression_params.push_back(3);
				//snprintf(filename_buffer, sizeof(filename_buffer), (root_ + "drone_raw_6/auto_dataset_bag_6_%06d.png").c_str(), counter);
				//std::string file_path = filename_buffer;
				//cv::imwrite(file_path, image, compression_params);
				//snprintf(filename_buffer, sizeof(filename_buffer), (root_ + "drone_bbox_6/auto_dataset_bag_6_%06d.png").c_str(), counter);
				//file_path = filename_buffer;
				//cv::imwrite(file_path, image_bbox, compression_params);


				//snprintf(annotation_buffer, sizeof(annotation_buffer), "auto_dataset_bag_6_%06d.png,0 %.10f %.10f %.10f %.10f", counter, (xmax+xmin)/im_width_/2, (ymax+ymin)/2/im_height_, (xmax-xmin)/im_width_, (ymax-ymin)/im_height_);
				//std::string annotation_string = annotation_buffer;
				//myfile_ << annotation_string << "\n";

				//std::cout << "Writing image to:" << file_path << std::endl;
				counter++;
				
			}
		}

		return;
	}
	

	bool ImageProcessor::someOdomBufferIsEmpty()
	{
		for(int i=0;i<(int)odometry_buffers.size();i++)
		{
			if(odometry_buffers[i].empty())
				return true;
		}
		return false;
	}

	cv::Point2d ImageProcessor::computeCentralPoint(Eigen::Isometry3d H_glasses_drone)
	{

		Eigen::Vector3d T_glasses_drone = H_glasses_drone.translation();

		// convert translation vector into cam frame
		T_glasses_drone = R_cam_world_*T_glasses_drone;

		// project onto camera frame
		Eigen::Vector3d projected_T = K*T_glasses_drone;
		cv::Point2d central_point(projected_T(0)/projected_T(2),
								  projected_T(1)/projected_T(2));
		return central_point;
	}

	std::vector<cv::Point2d> ImageProcessor::computeKeypointsCameraFrame(Eigen::Isometry3d H_glasses_drone)
	{
		std::vector<cv::Point2d> keypoints_camera_frame;

		for(int i=0;i<8;i++)
		{
			Eigen::Vector3d T_glasses_drone_edge = H_glasses_drone*keypoints_drone_frame[i];
			
			// convert translation vector into cam frame
			T_glasses_drone_edge = R_cam_world_*T_glasses_drone_edge; //CHECK HERE: R_CAM_WORLD ID TUNED WItH OFFSETS
			Eigen::Vector3d projective_T_edge = K*T_glasses_drone_edge;
			cv::Point2d point_edge(projective_T_edge(0)/projective_T_edge(2),
							  projective_T_edge(1)/projective_T_edge(2));
			keypoints_camera_frame.push_back(point_edge);
		}

		return keypoints_camera_frame;
	}

	//xmin, ymin, xmax, ymax
	std::vector<double> ImageProcessor::computeBboxLimits(std::vector<cv::Point2d> keypoints_camera_frame)
	{
		double bbox_xmin = keypoints_camera_frame[0].x;
		double bbox_ymin = keypoints_camera_frame[0].y;
		double bbox_xmax = keypoints_camera_frame[0].x;
		double bbox_ymax = keypoints_camera_frame[0].y;
		for (int i = 1; i < 8; i++) {
			bbox_xmin = std::min(bbox_xmin, keypoints_camera_frame[i].x);
			bbox_ymin = std::min(bbox_ymin, keypoints_camera_frame[i].y);
			bbox_xmax = std::max(bbox_xmax, keypoints_camera_frame[i].x);
			bbox_ymax = std::max(bbox_ymax, keypoints_camera_frame[i].y);
		}
		double margin_bbox = 20;
		bbox_xmin-=margin_bbox; bbox_ymin-= margin_bbox; bbox_xmax+=margin_bbox; bbox_ymax+=margin_bbox;  

		std::vector<double> bbox_limits;
		bbox_limits.push_back(bbox_xmin); bbox_limits.push_back(bbox_ymin); bbox_limits.push_back(bbox_xmax); bbox_limits.push_back(bbox_ymax); 
		return bbox_limits;
	}

	void ImageProcessor::drawDroneOnFullImage(cv::Mat image_full, cv::Point2d central_point, std::vector<cv::Point2d> keypoints_camera_frame, std::vector<double> bbox_limits)
	{
		double bbox_xmin = bbox_limits[0]; double bbox_ymin = bbox_limits[1]; double bbox_xmax = bbox_limits[2]; double bbox_ymax = bbox_limits[3]; 

		//Draw center of keypoints volume
		cv::circle(image_full, central_point, 1 , cv::Scalar(0, 255, 255), 2);

		//Draw keypoints volume
		cv::line(image_full, keypoints_camera_frame[0], keypoints_camera_frame[4], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[1], keypoints_camera_frame[5], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[2], keypoints_camera_frame[6], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[3], keypoints_camera_frame[7], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[0], keypoints_camera_frame[2], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[1], keypoints_camera_frame[3], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[4], keypoints_camera_frame[6], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[5], keypoints_camera_frame[7], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[0], keypoints_camera_frame[1], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[2], keypoints_camera_frame[3], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[4], keypoints_camera_frame[5], cv::Scalar(100, 100, 0), 2);
		cv::line(image_full, keypoints_camera_frame[6], keypoints_camera_frame[7], cv::Scalar(100, 100, 0), 2);

		//Draw bbox
		cv::rectangle(image_full, cv::Point2d(bbox_xmin, bbox_ymin), cv::Point2d(bbox_xmax, bbox_ymax), cv::Scalar(255,0,0), 2);
		
		//draw keypoints
		for (int i = 0; i < 8; i++) {
			if(i != 5 && i != 7)
				cv::circle(image_full, keypoints_camera_frame[i], 1 , cv::Scalar(255, 255, 0), 2); 
		}
		cv::circle(image_full, keypoints_camera_frame[5], 1 , cv::Scalar(255, 0, 255), 2); 
		cv::circle(image_full, keypoints_camera_frame[7], 1 , cv::Scalar(255, 0, 255), 2); 

	}

	void ImageProcessor::publishImage(image_transport::Publisher image_publisher, cv::Mat image, ros::Time ts_now)
	{
		cv_bridge::CvImage image_bridge;
		image_bridge.header.stamp = ts_now;
		image_bridge.image = image;
		image_bridge.encoding = sensor_msgs::image_encodings::BGR8;
		sensor_msgs::ImagePtr msg = image_bridge.toImageMsg();
		image_publisher.publish(msg);
	}

	std::vector<int> ImageProcessor::computeSquareBboxLimits(std::vector<double> bbox_limits)
	{
		double bbox_xmin = bbox_limits[0]; double bbox_ymin = bbox_limits[1]; double bbox_xmax = bbox_limits[2]; double bbox_ymax = bbox_limits[3]; 
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
		std::vector<int> square_bbox_limits;
		square_bbox_limits.push_back(square_bbox_xmin); square_bbox_limits.push_back(square_bbox_ymin); square_bbox_limits.push_back(square_bbox_xmax); square_bbox_limits.push_back(square_bbox_ymax); 
		return square_bbox_limits;
	}

	std::vector<cv::Point2d> ImageProcessor::computeKeypointsCropped(std::vector<cv::Point2d> keypoints_camera_frame, cv::Point2d square_coordinates_offset)
	{
		std::vector<cv::Point2d> keypoints_cropped;
		for(int i=0; i<8; i++)
		{
			cv::Point2d kpt_cropped;
			kpt_cropped = keypoints_camera_frame[i] - square_coordinates_offset;
			keypoints_cropped.push_back(kpt_cropped);
		}	
		return keypoints_cropped;
	}

	void ImageProcessor::drawDroneOnCroppedImage(cv::Mat image_cropped, std::vector<cv::Point2d> keypoints_cropped)
	{
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
		std::cout << counter << " frames have been saved to disk!" << std::endl;
		myfile_.close();
	}

}
