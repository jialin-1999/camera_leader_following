#include <opencv2/core.hpp>

#ifndef _REID_CONV_LINEAR_H_
#define _REID_CONV_LINEAR_H_

class ReidConvLinear
{
public:
	// Constructor
	ReidConvLinear(const cv::Size resolution, int resize_rate, int feature_num);
	
	// Initialize tracker
	void init(const cv::Rect &roi, cv::Mat image);
	
	// Get ConvNet feature from SSD network, and resize to the size camera_resolution/resize_rate;
	void getConvMat(std::vector<cv::Mat> &conv_feature);
	
	// Detect object in the current frame.
	float detect(cv::Rect roi, cv::Mat image);
	
	// Train re-id correlation filter
	void train(cv::Rect roi, cv::Mat image, float train_interp_factor);
	
	int tm_num = 0;
	
	double cosine(const cv::Mat &v1, const cv::Mat &v2);
	
protected:
	// Extract features
	cv::Mat getConvFeatures();
	
	// Evaluates a Linear kernel correlation between input images X and Y, which must both be MxN.
	// They must be pre-processed with a cosine window
	cv::Mat LinearCorrelation(cv::Mat x1, cv::Mat x2);
	
	cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);
	
	// Create Gaussian Peak(ridge regression target function). Called only in the first frame.
	cv::Mat createGaussianPeak(int sizey, int sizex);
	
	// Initialize Hanning window(cosine window) for FeatureMap. Called only in the first frame.
	cv::Mat createHanningMats(cv::Size tmpl_size, int feature_num);
	
	bool _convfeatures;
	std::vector<cv::Mat> _conv_feature;
	cv::Rect_<float> _roi;
	
	cv::Mat _alphaf;
	cv::Mat _prob;				// train target: gaussian
	cv::Mat _tmpl;
	cv::Mat _num;
	cv::Mat _den;
	
private:
	cv::Mat hann;
	cv::Size _tmpl_sz;
	
	float lambda; 			// regularization
	float padding; 			// extra area surrounding the target
	float output_sigma_factor; 	// bandwidth of gaussian target
	float kernel_sigma;
	
	int _resolution_h;
	int _resolution_w;
	int _resize_rate;
	int _feature_num;
	
	bool ini;
};


#endif