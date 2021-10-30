#ifndef _IDLA_H_
#define _IDLA_H_

#include "difference.h"
#include "input.h"
#include "multiclass_less.h"
#include "reinterpret.h"

#include <opencv2/core.hpp>
#include <dlib/dnn.h>

#include <string>

namespace idla{
	
typedef input_rgb_image_pair::input_type input_type;

template <
long num_filters,
long nr,
long nc,
int stride_y,
int stride_x,
typename SUBNET
>
using connp = dlib::add_layer<dlib::con_<num_filters,nr,nc,stride_y,stride_x,0,0>, SUBNET>;

template <long N, template <typename> class BN, long shape, long stride, typename SUBNET>
using block = dlib::relu<BN<connp<N, shape, shape, stride, stride, SUBNET>>>;   //param_list(kernel_num, layer_type, kernel_size, stride, SUBNET)

template <template <typename> class BN_CON, template <typename> class BN_FC>
using mod_idla = loss_multiclass_log_lr<dlib::fc<2,
dlib::relu<BN_FC<dlib::fc<500,reinterpret<2,
dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
block<25,BN_CON,5,5, // patch summary
dlib::relu<cross_neighborhood_differences<5,5,
dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,block<25,BN_CON,3,1,
dlib::max_pool<2,2,2,2,block<20,BN_CON,3,1,block<20,BN_CON,3,1,
input_rgb_image_pair
>>>>>>>>>>>>>>>>>;

using net_type = mod_idla<dlib::bn_con, dlib::bn_fc>;    // Training Net
using anet_type = mod_idla<dlib::affine, dlib::affine>;  // Testing Net

class Idla{

public:
	Idla(const std::string model_path);
	std::pair<float, float> Identify(const cv::Mat &person1, const cv::Mat &person2);

private:
	void Init();
	input_type cvt_Mat2Inputype(const cv::Mat &Person1, const cv::Mat &Person2);
	
	anet_type _net;
	dlib::softmax<anet_type::subnet_type> _snet; 	
	std::string _model_path;
};
	
}  // namespace

#endif