#include "reid_hog.hpp"
#include "fhog.hpp"
#include <opencv2/opencv.hpp>
#include "ffttools.hpp"
#include "recttools.hpp"
#include <iostream>

const int nClusters_Lab = 15;
float lab_data[nClusters_Lab][3] = {
	{161.317504, 127.223401, 128.609333},
	{142.922425, 128.666965, 127.532319},
	{67.879757, 127.721830, 135.903311},
	{92.705062, 129.965717, 137.399500},
	{120.172257, 128.279647, 127.036493},
	{195.470568, 127.857070, 129.345415},
	{41.257102, 130.059468, 132.675336},
	{12.014861, 129.480555, 127.064714},
	{226.567086, 127.567831, 136.345727},
	{154.664210, 131.676606, 156.481669},
	{121.180447, 137.020793, 153.433743},
	{87.042204, 137.211742, 98.614874},
	{113.809537, 106.577104, 157.818094},
	{81.083293, 170.051905, 148.904079},
	{45.015485, 138.543124, 102.402528}};
	
ReidHog::ReidHog()
{
	// Parameters equal in all cases
	lambda = 0.0001;				// regularization
	padding = 1.5;					// patch_size = 2.5 * target_img_size
	output_sigma_factor = 0.2;	    // bandwidth of gaussian target
	sigma = 0.3;					// gaussian kernel bandwidth
	interp_factor = 0.005;			// linear interpolation factor for adaptation
	
	// Feature map
	cell_size = 4;
	cell_sizeQ = cell_size*cell_size;
	_hogfeatures = true;
	_labfeatures = true;
	_labCentroids = cv::Mat(nClusters_Lab, 3, CV_32FC1, &lab_data);  // nClusters_Lab = 15;
	
	// templete, longer edge
	template_size = 96;
	_tmpl_sz = cv::Size2i(40,104);   // _tmpl.size() = [32, 96]
	size_patch[0] = 24;
	size_patch[1] = 8;
	if(_labfeatures)
		size_patch[2] = 46;
	else
		size_patch[2] = 31;
	
	re_size_patch[0] = size_patch[0];
	re_size_patch[1] = size_patch[1];
	re_size_patch[2] = size_patch[2];
	
	// adaptive templete update, from LMCF
	MAX_RESPONSE = 0.0;
	MEAM_ENERGY = 0.0;
	APCE = 0.0;
	APCE_NUM = 0;
	MAX_NUM = 0;
}

// Initialize tracker
void ReidHog::init(const cv::Rect &roi, cv::Mat image)
{
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);
	// create hanning window
	createHanningMats();   // size_patch[0], size_patch[1]
	// create gaussian labels
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);    		  // "y" in Ridge Regression
	// correlation filter coefficients
	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));  // real_ + i*imaginary_
	// init _tmpl
	bool is_feature_train = getFeatures(image, _tmpl, 1);
	// train
	train(_roi, image, 1.0);  	// intialize _tmpl
	std::cout << "--------------------REID init well --------------" << std::endl;
}

void ReidHog::detect(cv::Rect detect_rect, cv::Mat image, float &max_response, float &mean_energy)
{
	using namespace FFTTools;
	
	_roi = detect_rect;
	
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;
	
	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;
	
	// _tmpl = [384 x 46],	size_patch[0] = 16,	size_patch[1] = 24,	size_patch[2] = 46
	cv::Mat z;
	bool get_feature = ReidHog::getFeatures(image,z, false, 1.0f);
	if(get_feature == true){
		cv::Mat k = gaussianCorrelation(_tmpl, z);  // k.size() = [24,16]
		cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
		// f(z) = i_dfft(F(kxz)*F(alphaf)) --> real part of complex ---- formula(22) in paper
		
		// minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
		cv::Point2i pi;
		double pv, minv_d;   // peak value and minimum value
		cv::minMaxLoc(res, &minv_d, &pv, NULL, &pi);
		max_response = (float) pv;
		float minv = (float) minv;
		float sum_energy = 0.0;
		for(int i = 0; i <= res.rows-1; ++i){
			for(int j = 0; j <= res.cols-1; ++j){
				sum_energy += std::pow(res.at<float>(i,j)-minv,2);	
			}
		}
		mean_energy = sum_energy/(res.cols*res.rows);
		float delta_v = max_response - minv;
		APCE = delta_v/mean_energy;
		mean_energy = APCE;
		// 		std::cout << "APCE = "<<APCE<<std::endl;
		// cout << "!!!Re-id Max res = " << max_response << endl;
		// 	std::cout << "Max response = "<<peak_value<<std::endl;
		// 	std::cout << "Min response = "<<minv<<std::endl;
// 		cv::imshow("res", res);
// 		cv::waitKey(1);
	}
	// failed get feature
	else{
		std::cout << "--------Ri-id ------get feature failed!" << std::endl;
		max_response = 0.0;
		mean_energy = 0.0;
	}
}

void ReidHog::train(cv::Rect detect_rect, cv::Mat image, float train_interp_factor)
{
	using namespace FFTTools;
	
	_roi = detect_rect;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
	
	cv::Mat x;
	bool get_feature = ReidHog::getFeatures(image, x, 0);
	if(get_feature == true){
		cv::Mat k = gaussianCorrelation(x, x);
		
		cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
		
		_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
		_alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;
	}
}


// Obtain sub-window from image, with replication-padding and extract features
// if 1st frame create Hanning window, FeaturesMap = hann.mul(FeaturesMap);  
bool ReidHog::getFeatures(const cv::Mat &image, cv::Mat &FeaturesMap, bool inithann, float scale_adjust)
{
	cv::Rect extracted_roi;
	
	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;
	
	extracted_roi.width = _roi.x * padding;
	extracted_roi.height =  _roi.y * padding;
	
	// center roi with new size
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	
	if(extracted_roi.x <= 0 || extracted_roi.y <=0 || 
		extracted_roi.height <= 0 || extracted_roi.width <=0)
	{
		return false;
	}
	
	
	// 	cv::Mat FeaturesMap;
	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
	
	// _tmpl_sz = [40, 104] --> tmpl.size = [32, 96]
	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
		cv::resize(z, z, _tmpl_sz);
	}
	// 	cout << "---- re -----_tmpl_sz.height = " << _tmpl_sz.height << endl;
	// 	cout << "-----re ----- z.size = " << z.size()<< endl;
	// 	cout << "---- re -----gaussian = " << _prob.size() << endl;
	// HOG features
	if (_hogfeatures) {
		IplImage z_ipl = z;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&z_ipl, cell_size, &map);
		normalizeAndTruncate(map,0.2f);
		PCAFeatureMaps(map);
		size_patch[0] = map->sizeY;
		size_patch[1] = map->sizeX;
		size_patch[2] = map->numFeatures;
		
		// $$$: get feature well ??? !inithann
		if(size_patch[0] != re_size_patch[0] || size_patch[1] != re_size_patch[1])
		{
			free(map->map);
			delete map;
			map = NULL;
			return false;
		}
		
		FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
		FeaturesMap = FeaturesMap.t();
		
		freeFeatureMapObject(&map);
		
		// Lab features
		if (_labfeatures) {
			cv::Mat imgLab;
			cv::cvtColor(z, imgLab, CV_BGR2Lab);
			unsigned char *input = (unsigned char*)(imgLab.data);
			
			// Sparse output vector
			cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0]*size_patch[1], CV_32F, float(0));
			
			int cntCell = 0;
			// Iterate through each cell
			for (int cY = cell_size; cY < z.rows-cell_size; cY+=cell_size){
				for (int cX = cell_size; cX < z.cols-cell_size; cX+=cell_size){
					// Iterate through each pixel of cell (cX,cY)
					for(int y = cY; y < cY+cell_size; ++y){
						for(int x = cX; x < cX+cell_size; ++x){
							// Lab components for each pixel
							float l = (float)input[(z.cols * y + x) * 3];
							float a = (float)input[(z.cols * y + x) * 3 + 1];
							float b = (float)input[(z.cols * y + x) * 3 + 2];
							
							// Iterate trough each centroid
							float minDist = FLT_MAX;
							int minIdx = 0;
							float *inputCentroid = (float*)(_labCentroids.data);
							for(int k = 0; k < _labCentroids.rows; ++k){
								float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
								+ ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) )
								+ ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
								if(dist < minDist){
									minDist = dist;
									minIdx = k;
								}
							}
							// Store result at output
							outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
							// ((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ;
						}
					}
					cntCell++;
				}
			}
			// Update size_patch[2] and add features to FeaturesMap
			size_patch[2] += _labCentroids.rows;
			FeaturesMap.push_back(outputLab);
		}
	}
	
	
	FeaturesMap = hann.mul(FeaturesMap);
	
	return true;
}


// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat ReidHog::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
	using namespace FFTTools;
	cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
	// HOG features
	if (_hogfeatures) {
		cv::Mat caux;
		cv::Mat x1aux;
		cv::Mat x2aux;
		for (int i = 0; i < size_patch[2]; i++) {
			x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
			x1aux = x1aux.reshape(1, size_patch[0]);
			x2aux = x2.row(i).reshape(1, size_patch[0]);
			cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
			caux = fftd(caux, true);
			rearrange(caux);
			caux.convertTo(caux,CV_32F);
			c = c + real(caux);
		}
	}
	// Gray features
	else {
		cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
		c = fftd(c, true);
		rearrange(c);
		c = real(c);
	}
	cv::Mat d;
	cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 0, d);
	
	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
	return k;
}


// Create Gaussian Peak. Function called only in the first frame. Gaussian labels
cv::Mat ReidHog::createGaussianPeak(int sizey, int sizex)
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
void ReidHog::createHanningMats()
{
	cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
	cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0));
	
	for (int i = 0; i < hann1t.cols; i++)
		hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
	for (int i = 0; i < hann2t.rows; i++)
		hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
	
	cv::Mat hann2d = hann2t * hann1t;
	// HOG features
	if (_hogfeatures) {
		// reshape to 1-channel and 1-row
		cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
		// hann.size() = [W*H, 46] 46-rows and W*H-columns
		hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
		for (int i = 0; i < size_patch[2]; i++) {
			for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
				hann.at<float>(i,j) = hann1d.at<float>(0,j);
			}
		}
	}
	// Gray features
	else {  
		hann = hann2d;
	}
}
