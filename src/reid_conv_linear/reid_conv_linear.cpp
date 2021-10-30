// ------ Implement of class ReidConvLinear
#include "reid_conv_linear.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"

#include <iostream>

#include <opencv2/highgui.hpp>

using std::cout;
using std::endl;



ReidConvLinear::ReidConvLinear(const cv::Size resolution, int resize_rate, int feature_num)
{
	// Parameters equal in all cases
	lambda = 0.0001;				// regularization
	padding = 1.2;					// patch_size = 2.5 * target_img_size
	output_sigma_factor = 0.15;	    // bandwidth of gaussian target
	kernel_sigma = 0.2;
	
	ini = true;
	
	_resolution_h = resolution.height;
	_resolution_w = resolution.width;
	_resize_rate = resize_rate;
	_feature_num = feature_num;
	_tmpl_sz = cv::Size2i(16, 40);   // (14, 40)
}

// Initialize tracker
void ReidConvLinear::init(const cv::Rect &roi, cv::Mat image)
{
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);
	// create hanning window
	hann = createHanningMats(_tmpl_sz, _feature_num);
	// create gaussian labels
	_prob = createGaussianPeak(_tmpl_sz.height, _tmpl_sz.width);  // "y" in Ridge Regression
	// train 
	train(_roi, image, 1.0);  	// intialize _tmpl
	
	ini = false;
}


float ReidConvLinear::detect(cv::Rect roi, cv::Mat image)
{
	using namespace FFTTools;
	
	_roi = roi;
	
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;
	
	cv::Mat z = getConvFeatures();
	cv::Mat zf = FFTTools::fftd(z);
	
	// Compute AZ in the paper
	cv::Mat add_temp;  
	cv::reduce(FFTTools::complexMultiplication(_num, zf), add_temp, 0, CV_REDUCE_SUM);  
	cout << "here" << endl;
	cv::Mat res;
	cv::idft(FFTTools::complexDivisionReal(add_temp, _den + lambda), res, cv::DFT_REAL_OUTPUT);
// 	res = FFTTools::real(FFTTools::fftd(fftd(complexMultiplication(add_temp, _den + lambda), true)));
	res = res.reshape(0, _tmpl_sz.height);
	cout << "res = " << res.size() << endl;
	
	cv::imshow("res" , res);
	cv::waitKey(20);
// 	cv::Point2i pi;
	double maxv;
	cv::minMaxLoc(res, NULL, &maxv, NULL, NULL);
	
	return (float)maxv/_tmpl_sz.area();
}

void ReidConvLinear::train(cv::Rect roi, cv::Mat image, float train_interp_factor)
{
	_roi = roi;
	
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
	
	cv::Mat x =	getConvFeatures();
	
	cv::Mat xf = FFTTools::fftd(x);
	
	cv::Mat new_num;  
	cv::mulSpectrums(_prob, xf, new_num, 0, true);  
	cv::Mat new_den; 		
	cv::mulSpectrums(xf, xf, new_den, 0, true);
	cv::reduce(FFTTools::real(new_den), new_den, 0, CV_REDUCE_SUM);
	
	cout << "this" << endl;
	if(ini){
		_den = new_den;
		_num = new_num;
	}else{ 		// Get new A and new B
		_den = (1 - train_interp_factor) * _den + train_interp_factor * new_den;
		_num = (1 - train_interp_factor) * _num + train_interp_factor * new_num;
	}
}

// Calculate Linear correlation 
cv::Mat ReidConvLinear::LinearCorrelation(cv::Mat x1, cv::Mat x2) {
	using namespace FFTTools;
	cv::Mat c = cv::Mat( _tmpl_sz, CV_32F, cv::Scalar(0) );
	cv::Mat caux;
	cv::Mat x1aux;
	cv::Mat x2aux;
	for (int i = 0; i < _feature_num; i++) {
		x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
		x1aux = x1aux.reshape(1, _tmpl_sz.height);
		x2aux = x2.row(i).reshape(1, _tmpl_sz.height);
		
		cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
		caux = fftd(caux, true);
		rearrange(caux);
		caux.convertTo(caux,CV_32F);
		c = c + real(caux);
	}
	
	return c;
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat ReidConvLinear::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
	using namespace FFTTools;
	cv::Mat c = cv::Mat( _tmpl_sz, CV_32F, cv::Scalar(0) );
	
	// 	x1.convertTo(x1, CV_32FC1, 1.0/50);
	// 	x2.convertTo(x2, CV_32FC1, 1.0/50);
	// 		cout << "x1 = " << x1 <<endl<< endl;
	// 		cout << "x2 = " << x2 <<endl<< endl;
	cv::Mat caux;
	cv::Mat x1aux;
	cv::Mat x2aux;
	for (int i = 0; i < _feature_num; i++) {
		x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
		x1aux = x1aux.reshape(1, _tmpl_sz.height);
		x2aux = x2.row(i).reshape(1, _tmpl_sz.height);
		cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
		caux = fftd(caux, true);
		rearrange(caux);
		caux.convertTo(caux,CV_32F);
		c = c + real(caux);
	}
	// 	c = c.mul(1.0/255);
	// 	cout << "reid c = " << c << endl;
	
	cv::Mat d;
	cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (_tmpl_sz.area()*_feature_num), 0, d);  // / (_tmpl_sz.area()*_feature_num) 
	// 	cout << "reid d = " << d << endl;
	cv::Mat k;
	cv::exp((-d / (kernel_sigma * kernel_sigma)), k);
	// 	cout << "reid k2 = " << k << endl;
	// 	cout << "std::numeric_limits< float >::max() = " << std::numeric_limits< float >::max() << endl;
	// 	cv::waitKey(0);
	return k;
}

// Get FeatureMap [W, H ,C]
cv::Mat ReidConvLinear::getConvFeatures()
{
	std::vector<cv::Mat> vectorMat = _conv_feature;
	
	cv::Rect extracted_roi;
	
	float cx = (_roi.x + _roi.width / 2)  / _resize_rate;
	float cy = (_roi.y + _roi.height / 2) / _resize_rate;
	
	extracted_roi.width =  _roi.width * padding / _resize_rate;     // size of after resize_rate
	extracted_roi.height = _roi.height * padding / _resize_rate;
	
	extracted_roi.x = cx - extracted_roi.width / 2;                // left-up coordinate
	extracted_roi.y = cy - extracted_roi.height / 2;
	
	cv::Mat FeatureMap(cv::Mat(cv::Size( _tmpl_sz.area(), _feature_num), CV_32F, float(0)));
	
	for(int i = 0; i < vectorMat.size(); i++){
		cv::Mat z = RectTools::subwindow(vectorMat[i], extracted_roi, cv::BORDER_REPLICATE);
		cv::resize(z, z, _tmpl_sz);
		cv::Mat z_row = z.clone();
		z_row = z_row.reshape(1,1);
		z_row.copyTo(FeatureMap.row(i));
	}
	
	FeatureMap = hann.mul(FeatureMap);
	
	return FeatureMap;
}


// Get ConvNet feature from SSD network, and resize to the size camera_resolution/resize_rate;
void ReidConvLinear::getConvMat(std::vector<cv::Mat> &conv_feature)
{
	_conv_feature = conv_feature;
	// resize to 1/4 size of raw image in each channel; ==> [320, 180]
	for(int i = 0; i < _conv_feature.size(); i++){
		cv::resize(_conv_feature[i], _conv_feature[i], cv::Size(_resolution_w/_resize_rate,_resolution_h/_resize_rate));
	}
}


// Create Gaussian Peak. Function called only in the first frame. Gaussian labels
cv::Mat ReidConvLinear::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);
	
	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;
	
	float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);
	
	for (int i = 0; i < sizey; i++){
		for (int j = 0; j < sizex; j++)	{
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
		}
	}
	
	res = res.reshape(1, 1);
	cv::Mat prob = cv::repeat(res, _feature_num, 1); // ysf repeat totalSize times alone vertical axis
	
	return FFTTools::fftd(prob);
}


// Initialize Hanning window. Function called only in the first frame.
cv::Mat ReidConvLinear::createHanningMats(cv::Size tmpl_size, int feature_num)
{
	cv::Mat hann;
	cv::Mat hann1t = cv::Mat(cv::Size(tmpl_size.width,1), CV_32F, cv::Scalar(0));
	cv::Mat hann2t = cv::Mat(cv::Size(1,tmpl_size.height), CV_32F, cv::Scalar(0));
	
	for (int i = 0; i < hann1t.cols; i++)
		hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
	for (int i = 0; i < hann2t.rows; i++)
		hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
	
	cv::Mat hann2d = hann2t * hann1t;
	
	// reshape to 1-channel and 1-row
	cv::Mat hann1d = hann2d.reshape(1, 1); 
	// hann.size() = [W*H, 46] 46-rows and W*H-columns
	hann = cv::Mat(cv::Size(tmpl_size.area(), feature_num), CV_32F, cv::Scalar(0));
	for (int i = 0; i < feature_num; i++) {
		for (int j = 0; j<tmpl_size.area(); j++) {
			hann.at<float>(i,j) = hann1d.at<float>(0,j);
		}
	}
	
	return hann;
}

double ReidConvLinear::cosine(const cv::Mat &v1, const cv::Mat &v2)
{
	if(v1.size() != v2.size())
		return 0.0;
	
	double cosine = cv::sum(v1.mul(v2))[0]/(std::sqrt(cv::sum(v1.mul(v1))[0]) * std::sqrt(cv::sum(v2.mul(v2))[0]));
	
	return cosine;
}

