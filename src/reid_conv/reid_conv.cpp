// ------ Implement of class ReidConv
#include "reid_conv.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"

#include <iostream>

#include <opencv2/highgui.hpp>

using std::cout;
using std::endl;



ReidConv::ReidConv(const cv::Size resolution, int resize_rate, int feature_num)
{
	// Parameters equal in all cases
	lambda = 0.0001;				// regularization
	padding = 1.2;					// patch_size = 2.5 * target_img_size
	output_sigma_factor = 0.15;	    // bandwidth of gaussian target
	kernel_sigma = 0.2;
	
	_resolution_h = resolution.height;
	_resolution_w = resolution.width;
	_resize_rate = resize_rate;
	_feature_num = feature_num;
	_tmpl_sz = cv::Size2i(16, 40);   // (14, 40)
// 	_tmpl_sz = cv::Size2i(16, 36);
}

// Initialize tracker
void ReidConv::init(const cv::Rect &roi, cv::Mat image)
{
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);
	// create hanning window
	hann = createHanningMats(_tmpl_sz, _feature_num);
	// create gaussian labels
	_prob = createGaussianPeak(_tmpl_sz.height, _tmpl_sz.width);  // "y" in Ridge Regression
	// correlation filter coefficients
	_alphaf = cv::Mat(_tmpl_sz.height, _tmpl_sz.width, CV_32FC2, float(0));  // real_ + i*imaginary_
	// init _tmpl
	_tmpl = getConvFeatures();
	// train 
	train(_roi, image, 1.0);  	// intialize _tmpl
}

float ReidConv::detect(cv::Rect roi, cv::Mat image)
{
	using namespace FFTTools;
	
	_roi = roi;
	
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;
	
	cv::Mat z = getConvFeatures();
	cv::Mat k = LinearCorrelation(_tmpl, z);
	// 	cv::Mat k = gaussianCorrelation(_tmpl, z);
	cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
	
	cv::Point2i pi;
	double maxv_d;
	double minv_d;
	cv::minMaxLoc(res, &minv_d, &maxv_d, NULL, &pi);
	float maxv = (float) maxv_d;
	float minv = (float) minv_d;
	
// 	cv::imwrite("/home/pl/conv_feature_pic/all/raw/raw_img.jpg", image);

// 	cv::Mat gray = res.clone();
// 	gray.convertTo(gray, CV_8UC1, 255);
// 	cv::resize(gray, gray, cv::Size(0,0), 5, 5);
// 	cv::Mat color;
// 	cv::applyColorMap(gray, color, cv::COLORMAP_JET);
// 	cv::imwrite("/home/pl/conv_feature_pic/res/res_gray.jpg", gray);
// 	cv::imwrite("/home/pl/conv_feature_pic/res/res_JET.jpg", color);
// 	imshow("res_reid", color);
// 	imshow("res_reid_", gray);
// 	cv::waitKey(100);
	

//%% ENERGY
// 	cv::Rect extracted_roi;
// 	float cx = pi.x;
// 	float cy = pi.y;
// 	extracted_roi.width = 8;
// 	extracted_roi.height = 8;
// 	extracted_roi.x = cx - extracted_roi.width/2;
// 	extracted_roi.y = cy - extracted_roi.height/2;
// 	cv::Mat res_in = RectTools::subwindow(res, extracted_roi, cv::BORDER_CONSTANT);
// // 	res_in.reshape(0, 1);
// 	int num = 0;
// 	float sum_in = 0.0;
// 	for(int i = 0; i < res_in.rows; ++i){
// 		for(int j = 0; j < res_in.cols; ++j){
// 			if(res_in.at<float>(i, 0) > 0){
// 				sum_in += res_in.at<float>(j, i);
// 				++num;
// 			}
// 		}
// 	}
// 	cout << "sum_in = " << sum_in <<endl;
// %% APEC
// 	float delta_v = std::pow(maxv-minv,2);
// 	float sum_energy = 0.0;
// 	for(int i = 0; i <= res_in.rows-1; ++i){
// 		for(int j = 0; j <= res_in.cols-1; ++j){
// 			sum_energy += std::pow(res_in.at<float>(i,j)-minv,2);
// 		}
// 	}
// 	float mean_energy = sum_energy/res_in.size().area();

// 	float delta_v = std::pow(maxv-minv,2);
// 	float sum_energy = 0.0;
// 	for(int i = 0; i <= res.rows-1; ++i){
// 		for(int j = 0; j <= res.cols-1; ++j){
// 			sum_energy += std::pow(res.at<float>(i,j)-minv,2);
// 		}
// 	}
// 	float mean_energy = sum_energy/res.size().area();
// 
// 	float APCE = delta_v/mean_energy;
// 	std::cout << "-------------------------REID___APCE = "<<APCE<<std::endl;
// 	std::cout << "-------------------------REID___mean_energy = "<<mean_energy<<std::endl;
// 	std::cout << "-------------------------REID___min = "<<minv<<std::endl;
	
// 	double cosine_sim =	cosine(_tmpl, z);
// 	std::cout << "cosine sim = " << cosine_sim << std::endl;
	// cout << "!!!Re-id Max res = " << max_response << endl;
	// cv::imshow("_feature", _conv_feature[62]);
	// cv::waitKey(1);
	// cv::imshow("res", res);
// 	std::ostringstream out;  
// 	out << tm_num;  
// 	std::string tm_str = out.str(); 
// 	if(tm_num == 0)
// 		cv::imwrite("/home/pl/res/res_" + tm_str + ".jpg", res);
// 	tm_num = tm_num + 1;
// 	cv::waitKey(2);
	return (float)maxv;
}

void ReidConv::train(cv::Rect roi, cv::Mat image, float train_interp_factor)
{
	_roi = roi;
	
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	cv::Mat x =	getConvFeatures();
	cv::Mat k = LinearCorrelation(x, x);
// 	cv::Mat k = gaussianCorrelation(x, x);
	cv::Mat alphaf = FFTTools::complexDivision(_prob, (FFTTools::fftd(k) + lambda));
	
	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
	_alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;
}


// Calculate Linear correlation 
cv::Mat ReidConv::LinearCorrelation(cv::Mat x1, cv::Mat x2) {
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
cv::Mat ReidConv::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
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
cv::Mat ReidConv::getConvFeatures()
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
void ReidConv::getConvMat(std::vector<cv::Mat> &conv_feature)
{
	_conv_feature = conv_feature;
	// resize to 1/4 size of raw image in each channel; ==> [320, 180]
	for(int i = 0; i < _conv_feature.size(); i++){
		cv::resize(_conv_feature[i], _conv_feature[i], cv::Size(_resolution_w/_resize_rate,_resolution_h/_resize_rate));
	}
	
// 	for(int i = 0; i < _feature_num; ++i){
// 		std::ostringstream ss;
// 		ss << i+1;
// 		cv::imwrite("/home/pl/conv_feature_pic/gray6/" + ss.str() + "_gray_conv2_2.jpg", conv_feature[i]);
// 		cv::Mat gray = _conv_feature[i];
// 		gray.convertTo(gray, CV_8UC1);
// 		cv::Mat color;
// 		cv::applyColorMap(gray, color, cv::COLORMAP_JET);
// 		cv::imwrite("/home/pl/conv_feature_pic/all/jet/" + ss.str() + "_JET_conv2_2.jpg", color);
// 		imshow("feature6", color);
// 		cv::waitKey(10);
// 	}
	
}


// Create Gaussian Peak. Function called only in the first frame. Gaussian labels
cv::Mat ReidConv::createGaussianPeak(int sizey, int sizex)
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
	
	return FFTTools::fftd(res);
}


// Initialize Hanning window. Function called only in the first frame.
cv::Mat ReidConv::createHanningMats(cv::Size tmpl_size, int feature_num)
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

double ReidConv::cosine(const cv::Mat &v1, const cv::Mat &v2)
{
	if(v1.size() != v2.size())
		return 0.0;
	
	double cosine = cv::sum(v1.mul(v2))[0]/(std::sqrt(cv::sum(v1.mul(v1))[0]) * std::sqrt(cv::sum(v2.mul(v2))[0]));
	
	return cosine;
}

