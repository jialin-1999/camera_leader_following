// long-term re-id using HoG+LAB features

#ifndef _REID_H_
#define _REID_H_

#include <opencv2/core.hpp>


	
class ReidHog
{
public:
	// Constructor
	ReidHog();
	
	// Initialize tracker
	void init(const cv::Rect &roi, cv::Mat image);
	
	// Detect object in the current frame.
	void detect(cv::Rect detect_rect, cv::Mat image, float &max_response, float &mean_energy);
	
	// Train re-id correlation filter
	void train(cv::Rect detect_rect, cv::Mat image, float train_interp_factor);
	
	float interp_factor; 	// linear interpolation factor for adaptation
	float sigma; 			// gaussian kernel bandwidth
	float lambda; 			// regularization
	int cell_size; 			// HOG cell size
	int cell_sizeQ; 		// cell size^2, to avoid repeated operations
	float padding; 			// extra area surrounding the target
	float output_sigma_factor; 	// bandwidth of gaussian target
	int template_size; 		// template size
	
	float MAX_RESPONSE;
	float MEAM_ENERGY;
	float APCE;
	int APCE_NUM;
	int MAX_NUM;
	
	
protected:
	
	
	// Obtain sub-window from image, with replication-padding and extract features
	bool getFeatures(const cv::Mat & image,cv::Mat &FeatureMap, bool inithann, float scale_adjust = 1.0f);
	
	// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts 
	// between input images X and Y, which must both be MxN. 
	// They must also be periodic (ie., pre-processed with a cosine window).
	cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);
	
	// Create Gaussian Peak. Function called only in the first frame.
	cv::Mat createGaussianPeak(int sizey, int sizex);
	
	// Initialize Hanning window. Function called only in the first frame.
	void createHanningMats();
	
	cv::Mat _alphaf;
	cv::Mat _prob;				// train target: gaussian
	cv::Mat _tmpl;
	cv::Mat _num;
	cv::Mat _den;
	cv::Mat _labCentroids;
	
private:
	int size_patch[3];  // feature map size: 0: map->Y(height) 1: map->X(width) 2: map->deep(channel)
	cv::Mat hann;
	cv::Size _tmpl_sz;
	float _scale;
	int _gaussian_size;
	bool _hogfeatures;
	bool _labfeatures;  
	
	int re_size_patch[3];

protected:
	cv::Rect_<float> _roi;
	
};


#endif