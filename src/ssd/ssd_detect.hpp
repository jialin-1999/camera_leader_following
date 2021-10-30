/*Indication*/
// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
/* */
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
using namespace caffe;

class Detector
{
public:
Detector(const std::string& model_file,
		 const std::string& weights_file,
		 const std::string& mean_file,
		 const std::string& mean_value);

std::vector< std::vector<float> > Detect(const cv::Mat& img_rgb);

std::vector<cv::Mat> GetFeature(); // get feature for tracker


private:
void SetMean(const std::string& mean_file, const std::string& mean_value);

void WrapInputLayer(std::vector<cv::Mat>* input_channels);

void Preprocess(const cv::Mat& img_rgb,
                std::vector<cv::Mat>* input_channels);

private:
boost::shared_ptr<caffe::Net<float> > net_;
cv::Size input_geometry_;
int num_channels_;
cv::Mat mean_;

std::vector<cv::Mat> _feature_map; //restore feature for tracker


};
