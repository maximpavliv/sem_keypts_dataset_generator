#include "dataset_generator/dataset_generator.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_processor");
    ros::NodeHandle nh;
    ros::NodeHandle nh_priv("~");
    image_processor::ImageProcessor IP(nh, nh_priv);

    ros::spin();
}
