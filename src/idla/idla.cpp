#include "idla.hpp"
#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include <opencv2/highgui.hpp>
#include <dlib/gui_widgets.h>
namespace idla{

Idla::Idla(const std::string model_path)
{
	_model_path = model_path;
	Init();
}

void Idla::Init()
{
	dlib::deserialize(_model_path) >> _net;
	_snet.subnet() = _net.subnet();
	std::cout << "----- # ####################################### # ------" << std::endl;
	std::cout << "-----   Idla person re-id has initialized well!   ------" << std::endl;
	std::cout << "----- # ####################################### # ------" << std::endl;
}


std::pair<float, float> Idla::Identify(const cv::Mat &person1, const cv::Mat &person2)
{
	input_type img_pair = cvt_Mat2Inputype(person1, person2);
	dlib::matrix<float> output = dlib::mat(_snet(img_pair));
	float different = output(0, 0);
	float same = output(0, 1);
	
	return std::make_pair(different, same);
}


input_type Idla::cvt_Mat2Inputype(const cv::Mat &Person1, const cv::Mat &Person2)
{
	cv::Mat person1 = Person1;
	cv::Mat person2 = Person2;
	
	cv::resize(person1, person1, cv::Size(60, 160));
	cv::resize(person2, person2, cv::Size(60, 160));
	cv::cvtColor(person1, person1, CV_BGR2RGB);
	cv::cvtColor(person2, person2, CV_BGR2RGB);
	dlib::matrix<dlib::rgb_pixel>  person_1, person_2;
	dlib::assign_image(person_1, dlib::cv_image<dlib::rgb_pixel>(person1));
	dlib::assign_image(person_2, dlib::cv_image<dlib::rgb_pixel>(person2));
	input_type img_pair(&person_1, &person_2);

	return img_pair;
}

}