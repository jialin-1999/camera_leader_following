/*

Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame

 */

const int nClusters = 15;
float data[nClusters][3] = {
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
	
#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
// #include "labdata.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/highgui.hpp>
#endif

using std::cout;
using std::endl;

KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{
	// Parameters equal in all cases
	lambda = 0.0001;				// regularization
	padding = 2.3;					// patch_size = 2.5 * target_img_size
	padding_h = 1.5;    			// WARNING
	padding_w = 2.5;
	output_sigma_factor = 0.1;	    // bandwidth of gaussian target
	sigma = 0.4;					// gaussian kernel bandwidth
	interp_factor = 0.01;			// linear interpolation factor for adaptation
	cell_size = 4;
	cell_sizeQ = cell_size*cell_size;
	_hogfeatures = true;
	_labfeatures = true;
	_labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);  // nClusters = 15;
	
	if (multiscale) { // multiscale
		template_size = 96;
		//scale parameters initial
		scale_padding = 1.0;
		scale_step = 1.05;
		scale_sigma_factor = 0.25;
		n_scales = 33;
		scale_lr = 0.025;         // scale templete learning rate
		scale_max_area = 512;
		currentScaleFactor = 1;
		scale_lambda = 0.01;
		if (!fixed_window) {  	  // templete size is fixed
			fixed_window = true;  // Multiscale does not support non-fixed window
		}
	}
	else if (fixed_window) {  	  // fit correction without multiscale
		template_size = 96;
		scale_step = 1;
	}
	else {
		template_size = 1;
		scale_step = 1;
	}
	
	// adaptive templete update, from LMCF
	MAX_RESPONSE = 0.0;
	APCE = 0.0;
	
	// init
	init_success = true;
}

// Initialize tracker
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);

	getFeatures(image, _tmpl, 1);
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);    		 // "y" in Ridge Regression
	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0)); // complex: real + i*imaginary
	// size_patch[0] = map->sizeY --height; size_patch[1] = map->sizeX --width;
	dsstInit(roi, image);
	train(_tmpl, 1.0); 		// train with initial frame
}

// Initialization for scales
void KCFTracker::dsstInit(const cv::Rect &roi, cv::Mat image)
{
	// The initial size for adjusting
	base_width = roi.width;
	base_height = roi.height;
	
	// Guassian peak for scales (after fft)
	ysf = computeYsf();                       // create gaussian label
	sf_hann = createHanningMatsForScale();	  // create hanning window
	
	// Get all scale changing rate
	// scaleFactors[33] = 1.05^{16,15,14,13,...,1,0,-1,...,-16} ; 1.05^16 = 1.373   1.05^-16 = 0.458
	scaleFactors = new float[n_scales];
	float ceilS = std::ceil(n_scales / 2.0f); // round up to an integer
	for(int i = 0 ; i < n_scales; i++){
		scaleFactors[i] = std::pow(scale_step, ceilS - i - 1);  // scale_step = 1.05
	}
	
	// Get the scaling rate for compressing to the model size
	// if initial ROI too large, resize scale_model_height & scale_model_width
	float scale_model_factor = 1;
	if(base_width * base_height > scale_max_area){  // limit the max scale model area = 512;
		scale_model_factor = std::sqrt(scale_max_area / (float)(base_width * base_height));
	}
	scale_model_width = (int)(base_width * scale_model_factor); 
	scale_model_height = (int)(base_height * scale_model_factor);
	
	// ??????? Compute min and max scaling rate
	min_scale_factor = std::pow(scale_step, std::ceil(std::log((std::max(5 / (float) base_width, 5 / (float) base_height) * (1 + scale_padding))) / 0.0086));
	max_scale_factor = std::pow(scale_step, std::floor(std::log(std::min(image.rows / (float) base_height, image.cols / (float) base_width)) / 0.0086));
	// min_scale_factor = 5.03512e-11  	max_scale_factor = 1.62889
	
	train_scale(image, true);
}
 
// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image, bool is_train)
{
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;
	
	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;
	
	float peak_value;
	cv::Mat featuremap_detect;
	int is_feature = getFeatures(image,featuremap_detect, 0);
	cv::Point2f res;
	if(is_feature != -1)
		res = detect(_tmpl, featuremap_detect, peak_value);
	else
		res = cv::Point2f(float(0.0001),float(0.0001));
	
	// BUG Adjust target bbox by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale * currentScaleFactor);
	_roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale * currentScaleFactor);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
	
	// Update scale
	cv::Point2i scale_pi = detect_scale(image);
	currentScaleFactor = currentScaleFactor * scaleFactors[scale_pi.x];
	// Limit Max bbox size and Min bbox size
	if(currentScaleFactor < min_scale_factor)
		currentScaleFactor = min_scale_factor;
	if(currentScaleFactor > max_scale_factor)
		currentScaleFactor = max_scale_factor;

	// Train new _tmpl and _alphaf
	if(is_train){
		// Train scale filter and Update _roi
		train_scale(image);
		if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
		if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;
		if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
		if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
		assert(_roi.width >= 0 && _roi.height >= 0);
		// Train translation filter		
		cv::Mat featuremap_train;
		int is_feature_train = getFeatures(image, featuremap_train, 0);
		if(is_feature_train != -1)
			train(featuremap_train, interp_factor);
	}

	return _roi;
}

// Detect object in the current frame. return the translation coordinate.
//  x.size() = [384,46],  z.size() = [384,46], 31-dim hog + 15-dim lab
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;

	// f(z) = i_dfft(F(kxz)*F(alphaf)) --> real part of complex ---- formula(22) in paper
    cv::Mat k = gaussianCorrelation(x, z);  // k.size() = [24,16]
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

	// minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
	double minv_d;
    cv::minMaxLoc(res, &minv_d, &pv, NULL, &pi);
    peak_value = (float) pv;
	float minv = (float) minv;

	//subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);
	// point: pi(pi.x, pi.y)
    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }
    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    // center of cv::Mat res --> (cx,cy) is the position of target in the last frame 
	p.x -= (res.cols) / 2;   // p.x = p.x - cx,
	p.y -= (res.rows) / 2;	 // p.y = p.y - cy,
	
	// Compute APCE
	float delta_v = std::pow(peak_value-minv,2);
	float sum_energy = 0.0;
	for(int i = 0; i <= res.rows-1; ++i){
		for(int j = 0; j <= res.cols-1; ++j){
			sum_energy += std::pow(res.at<float>(i,j)-minv,2);	
		}
	}
	float mean_energy = sum_energy/(res.cols*res.rows);
	APCE = delta_v/mean_energy;
	MAX_RESPONSE = peak_value;

	// BUG  When peak_value == 0, p will be very big number
	if(peak_value > 0 && peak_value < 1000)  	// p is the translation coordinate 
		return p;  
	else
		return cv::Point2f(float(0.00001), float(0.00001));
}

// Detect the new scaling rate
cv::Point2i KCFTracker::detect_scale(cv::Mat image)
{
	cv::Mat xsf;
	int has_get = KCFTracker::get_scale_sample(image, xsf);  // xsf.size() = [33, totalSize]

	if(has_get == -1 ){
		return cv::Point2i(int(0),int(0));
	}

	// Compute AZ in the paper
	cv::Mat add_temp;  
	cv::reduce(FFTTools::complexMultiplication(_sf_num, xsf), add_temp, 0, CV_REDUCE_SUM);  // [33, totalSize] -> add_temp[33, 1]

	// Compute the final y; size = [33,1]; $$ fomula: y = (AZ)/(B + lamada) $$
	cv::Mat scale_response;
	cv::idft(FFTTools::complexDivisionReal(add_temp, (_sf_den + scale_lambda)), scale_response, cv::DFT_REAL_OUTPUT);
	
	// Get the max point as the final scaling rate
	cv::Point2i pi;
	double pv;
	cv::minMaxLoc(scale_response, NULL, &pv, NULL, &pi);

	return pi;
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;
}

// Train method for scaling
void KCFTracker::train_scale(cv::Mat image, bool ini)
{
	cv::Mat xsf;  // sample xsf.size() = [33,(248)], 33 column vectors[33, (248)]
	int is_get = get_scale_sample(image, xsf);  
	
	// failed to get scale sample
	if(is_get == -1){
		if(ini)
			init_success = false;
	}else{ 
		// Adjust ysf to the same size as xsf in the first time
		if(ini){
			int totalSize = xsf.rows;			 //  totalSize = w*h
			ysf = cv::repeat(ysf, totalSize, 1); // ysf repeat totalSize times alone vertical axis
		} 
		// Get new GF in the paper (delta A), numerator
		cv::Mat new_sf_num;  	// [33, totalSize]
		cv::mulSpectrums(ysf, xsf, new_sf_num, 0, true);  // Performs the per-element multiplication correlation type 
	
		// Get Sigma{FF} in the paper (delta B), denominator
		cv::Mat new_sf_den; 		// [33, totalSize] 
		cv::mulSpectrums(xsf, xsf, new_sf_den, 0, true);
		cv::reduce(FFTTools::real(new_sf_den), new_sf_den, 0, CV_REDUCE_SUM); // [33, totalSize]->[33, 1], CV_REDUCE_SUM--> output sum of all matrix row(here)/column 
				
		// after: xsf[33, 284] , ysf[33, 284] , new_sf_num[33, 248] ,  new_sf_den[33, 1] 
		if(ini){
			_sf_den = new_sf_den;
			_sf_num = new_sf_num;
		}else{ 		// Get new A and new B
			_sf_den = (1 - scale_lr) * _sf_den + scale_lr * new_sf_den;
			_sf_num = (1 - scale_lr) * _sf_num + scale_lr * new_sf_num;
		}
		// Update _roi according to the new scale
		update_roi();
	}
}	



// Update the ROI size after training
void KCFTracker::update_roi()
{
	// Compute new center
	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;
	
	// Recompute the ROI left-upper point and size
	_roi.width = base_width * currentScaleFactor;
	_roi.height = base_height * currentScaleFactor;
	_roi.x = cx - _roi.width / 2.0f;
	_roi.y = cy - _roi.height / 2.0f;
}


// Obtain sub-window from image, with replication-padding and extract features
// if 1st frame create Hanning window, FeaturesMap = hann.mul(FeaturesMap);  
int KCFTracker::getFeatures(const cv::Mat & image, cv::Mat &FeaturesMap, bool inithann)
{
	cv::Rect extracted_roi;
	
	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;
	
	if (inithann) {
		//int padded_w = _roi.width * padding;
		//int padded_h = _roi.height * padding;
		int padded_w = _roi.width * padding_w;
		int padded_h = _roi.height * padding_h;
		if (template_size > 1) {  // Fit largest dimension to the given template size
			if (padded_w >= padded_h)  //fit to width
				_scale = padded_w / (float) template_size;
			else
				_scale = padded_h / (float) template_size;
			
			_tmpl_sz.width = padded_w / _scale;
			_tmpl_sz.height = padded_h / _scale;
		}
		else {  // No template size given, use ROI size
			_tmpl_sz.width = padded_w;
			_tmpl_sz.height = padded_h;
			_scale = 1;
		}
		
		if (_hogfeatures) {
			// Round to cell size and also make it even
			_tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
			_tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
		}else {  
			// Make number of pixels even (helps with some logic involving half-dimensions)
			_tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
			_tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
		}
	}
	
	extracted_roi.width =  _scale * _tmpl_sz.width * currentScaleFactor;
	extracted_roi.height = _scale * _tmpl_sz.height * currentScaleFactor;

	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	
	if(extracted_roi.width == 0 || extracted_roi.height ==0	)
		return -1;

	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
	
	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
		cv::resize(z, z, _tmpl_sz);
	}
	
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
		
		// BUG identify whether get HoG feture well?
		if(!inithann && (size_patch[0] != re_size_patch[0] || size_patch[1] != re_size_patch[1])){
			free(map->map);
			delete map;
			map = NULL;
			return -1;
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
							// Store result at output, distance most near to one of the LAB data, 
							// count +1 (+1.0 / cell_sizeQ)
							outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
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
	// Raw pixel gray feature
	else {  
		FeaturesMap = RectTools::getGrayImage(z);
		FeaturesMap -= (float) 0.5;  // In Paper;
		size_patch[0] = z.rows;
		size_patch[1] = z.cols;
		size_patch[2] = 1;
	}
	
	if (inithann) {
		re_size_patch[0] = size_patch[0];
		re_size_patch[1] = size_patch[1];
		re_size_patch[2] = size_patch[2];
		createHanningMats();
	}
	// FeatureMap is a numfeature-row vector, multilied by hanning window function, hann.size() = [W*H,46]
	FeaturesMap = hann.mul(FeaturesMap); 
	
	return 0;
}

// Compute the F^l in the paper
int KCFTracker::get_scale_sample(const cv::Mat & image, cv::Mat &xsf)
{
	CvLSVMFeatureMapCaskade *map[n_scales]; // temporarily store FHOG result
	int totalSize; 							// numbers of features
	
	bool is_extractImage = true;
	for(int i = 0; i < n_scales; i++){
		// Size of subwindow waiting to be detect, base_height is the bbox size in 1st frame
		float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
		float patch_height = base_height * scaleFactors[i] * currentScaleFactor;
		
		float cx = _roi.x + _roi.width / 2.0f;
		float cy = _roi.y + _roi.height / 2.0f;
		
		// Get the subwindow
		cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
		
		// BUG im_patch extract faild, target go to the edge of image
		if(im_patch.cols <= 5 || im_patch.rows <= 5){
			is_extractImage = false;
			break;
		}
		
		// Scaling the subwindow
		cv::Mat im_patch_resized;
		if(scale_model_width > im_patch.cols)  				// CV_INTER_LINEAR = 1,
			cv::resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 1);
		else  												// CV_INTER_AREA  = 3,
			cv::resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 3);

		// Compute the fHoG features for the subwindow,
		IplImage im_ipl = im_patch_resized;
		getFeatureMaps(&im_ipl, cell_size, &map[i]);
		normalizeAndTruncate(map[i], 0.2f);
		PCAFeatureMaps(map[i]);
		
		if(i == 0){
			totalSize = map[i]->numFeatures * map[i]->sizeX * map[i]->sizeY;

			// BUG Sometimes after compute HoG feture, map[i]->sizeX = 0
			if(totalSize == 0){
				break;
			}else{			
				xsf = cv::Mat(cv::Size(n_scales,totalSize), CV_32F, float(0));  // xsf.size() = [33,totalSize]
			}
		}
		// FHOG feature vector size() = [1, totalSize], row vector
		cv::Mat FeaturesMap = cv::Mat(cv::Size(1, totalSize), CV_32F, map[i]->map);

		// Multiply the FHOG results by hanning window and copy to the output
		float mul = sf_hann.at<float > (0, i);
		FeaturesMap = mul * FeaturesMap;
		FeaturesMap.copyTo(xsf.col(i));
	}
	
	// After 33 times coparate in feature vector, has getten xsf[33,totalSize] --> 33 column vector 
	if(totalSize == 0 || is_extractImage == false){
		cout <<"  -------- Failed Get Feature For Scale Filter!" << endl;
		return -1;
	}
	
	// Free the temp variables
	for(int i = 0; i < n_scales; i++)
		freeFeatureMapObject(&map[i]);

	// Do fft to the FHOG features row by row
	xsf = FFTTools::fftd(xsf, 0, 1);

	return 0;
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
	using namespace FFTTools;
	cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );

// 	cout << "reid x1 = " << x1 << endl;
// 	cout << "reid x2 = " << x2 << endl;
	
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
// 	cout << "reid c = " << c << endl;
// 	cout << "reid d = " << d << endl;
	
	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
// 	cout << "reid k = " << k << endl;
// 	cv::waitKey(0);
	return k;
}

// Create Gaussian Peak. Function called only in the first frame. Gaussian labels
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);
	
	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;
	
	//float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
	float output_sigma = std::sqrt((float) sizex * sizey) / padding_w * output_sigma_factor;
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
void KCFTracker::createHanningMats()
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
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug

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

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}


// Compute the FFT Guassian Peak for scaling
cv::Mat KCFTracker::computeYsf()
{
	cv::Mat res(cv::Size(n_scales, 1), CV_32F, float(0));  // res.size() = [1,33] -> 33-dim column vector
	
	float scale_sigma2 = std::sqrt((float)n_scales) * scale_sigma_factor; // get sigma
    scale_sigma2 = scale_sigma2 * scale_sigma2;  // sigma2 = 2.0625
    
	float ceilS = std::ceil(n_scales / 2.0f);
	
	// create 1-dim gaussian peak "res"
    for(int i = 0; i < n_scales; i++){
		res.at<float>(0,i) = std::exp(- 0.5 * std::pow(i + 1- ceilS, 2) / scale_sigma2);
    }

    return FFTTools::fftd(res);
}

// Compute the hanning window for scaling
cv::Mat KCFTracker::createHanningMatsForScale()
{
	cv::Mat hann_s = cv::Mat(cv::Size(n_scales, 1), CV_32F, cv::Scalar(0));
	
	for (int i = 0; i < hann_s.cols; i++){
		hann_s.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann_s.cols - 1)));
	}
	
	return hann_s;
}




