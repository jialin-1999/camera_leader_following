#include "ssd_detect.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>

#include <utility>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <sstream>

#include <caffe/caffe.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value)
{
	#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
	#else
		Caffe::set_mode(Caffe::GPU);
	#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));    // Read net structure, initialize network
	net_->CopyTrainedLayersFrom(weights_file);       // read weights of Network
	
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
	
	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();    // get the channels of imput
	CHECK(num_channels_ == 3 || num_channels_ == 1) 
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());  // get size of input image
	/* Load the binaryproto mean file. */
	SetMean(mean_file, mean_value);  // initialize mean file
	// end all initialization
}

// blob[4]: [0]->num num of images [1]-> channel [2]-> width [3]->height
// input image, return result , each vector present a result(including position and class confidence)
std::vector<vector<float> > Detector::Detect(const cv::Mat& img_rgb)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();     // Reshape the Network

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);   // bound the input of network with input_channels %%

	Preprocess(img_rgb, &input_channels);  // preprocessing

	net_->Forward();    // forward propagation of network

	
	// WARNING
	/* ---------------------  extract ConvNet feature -------------------- */
	string blob_name = "conv2_2";
	CHECK(net_->has_blob(blob_name))
	<< "Unknown feature blob name " << blob_name << " in the network " << "SSD_300x300_VGG16";
	boost::shared_ptr<Blob<float> >  feature_blobs = net_->blob_by_name(blob_name);
	int blob_channels = feature_blobs->channels();	
	_feature_map.resize(blob_channels);
	const float *pstart = feature_blobs->cpu_data();
	for(int i = 0; i < blob_channels; i++){
		cv::Mat feature_tmp(feature_blobs->height(), feature_blobs->width(), 
							CV_32FC1, const_cast<float*>(pstart));
		pstart += feature_blobs->height() * feature_blobs->width();
		_feature_map[i] = feature_tmp;
	}
	/* ---------------------  ######################### -------------------- */
	
	
	
	/* Copy the output layer to a std::vector */
	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();   // read output information
	const int num_det = result_blob->height();
	vector<vector<float> > detections;
	for(int k = 0; k < num_det; ++k){
		if (result[0] == -1){   // -1 present the background
			// Skip invalid detection.
			result += 7;
			continue;
		}
		vector<float> detection(result, result + 7);
		detections.push_back(detection);
		result += 7;
	}
	return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value)
{
	cv::Scalar channel_mean;
	if (!mean_file.empty()){
		CHECK(mean_value.empty()) 
			<<"Cannot specify mean_file and mean_value at the same time";
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

		/* Convert from BlobProto to Blob<float> */
		Blob<float> mean_blob;
		mean_blob.FromProto(blob_proto);
		CHECK_EQ(mean_blob.channels(), num_channels_)
			<< "Number of channels of mean file doesn't match input layer.";

		/* The format of the mean file is planar 32-bit float BGR or grayscale. */
		std::vector<cv::Mat> channels;
		float* data = mean_blob.mutable_cpu_data();
		for (int i = 0; i < num_channels_; ++i){
			/* Extract an individual channel. */
			cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
			channels.push_back(channel);
			data += mean_blob.height() * mean_blob.width();
		}

		/* Merge the separate channels into a single image. */
		cv::Mat mean;
		cv::merge(channels, mean);

		/* Compute the global mean pixel value and create a mean image filled with this value. */
		channel_mean = cv::mean(mean);
		mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
	}
	if (!mean_value.empty()){
		CHECK(mean_file.empty()) 
			<< "Cannot specify mean_file and mean_value at the same time";
		stringstream ss(mean_value);
		vector<float> values;
		string item;
		while (getline(ss, item, ',')){
			float value = std::atof(item.c_str());
			values.push_back(value);
		}
		CHECK(values.size() == 1 || values.size() == num_channels_) 
			<<"Specify either 1 mean_value or as many as channels: " << num_channels_;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < num_channels_; ++i){
			/* Extract an individual channel. */
			cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(values[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, mean_);
	}
}

/* Wrap the input layer of the network in separate cv::Mat objects (one per channel). 
 * This way we save one memcpy operation and we don't need to rely on cudaMemcpy2D. 
 * The last preprocessing operation will write the separate channels directly to the input layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) // bind input_channels and input of net together
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i){
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

/* Convert the input image to the input image format of the network. */
void Detector::Preprocess(const cv::Mat& img_rgb, std::vector<cv::Mat>* input_channels)
{
	cv::Mat sample;
	if (img_rgb.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img_rgb, sample, cv::COLOR_BGR2GRAY);
	else if (img_rgb.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img_rgb, sample, cv::COLOR_BGRA2GRAY);
	else if (img_rgb.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img_rgb, sample, cv::COLOR_BGRA2BGR);
	else if (img_rgb.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img_rgb, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img_rgb;
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;
	//cv::imshow("resized",sample_resized);
	//cv::waitKey(1);
	cv::Mat sample_float;
	if (num_channels_ == 3){
		sample_resized.convertTo(sample_float, CV_32FC3);
	}
	else{
		sample_resized.convertTo(sample_float, CV_32FC1);
	}

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/*  This operation will write the separate BGR planes directly to the input layer of the network, 
	 *  because it is wrapped by the cv::Mat objects in input_channels. 	 */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}




std::vector<cv::Mat> Detector::GetFeature()
{
	return _feature_map;
}

