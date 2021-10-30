#include "opencv2/opencv.hpp"
#include "sl/defines.hpp"
#include <limits.h>

// overlap in x-axis
bool xxOverlap(const cv::Rect& box1, const cv::Rect& box2)
{
	if(box1.x > box2.x + box2.width) 
		return false; 
	else if(box1.x + box1.width < box2.x)
		return false; 
	else
		return true;
}


float bbOverlap(const cv::Rect& box1, const cv::Rect& box2)
{
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }
	float colInt =  std::min(box1.x+box1.width,box2.x+box2.width) - std::max(box1.x, box2.x);
	float rowInt =  std::min(box1.y+box1.height,box2.y+box2.height) - std::max(box1.y,box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}



float bbAreaRatio(const cv::Rect& box1, const cv::Rect& box2)
{
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return area1 / area2;
}

ushort computeDistance(const cv::Mat &depth_map, const cv::Rect target_box)
{
	// 	clip the bbox
	int x1 = std::max(2, target_box.x);
	int y1 = std::max(2, target_box.y);
	int x2 = std::min(target_box.x + target_box.width , depth_map.cols-2);
	int y2 = std::min(target_box.y + target_box.height, depth_map.rows-2);
	
	if(x2 - x1 <= 0 || y2 - y1 <= 0)
		return 0;

	// compute depth mean
	unsigned long depth_sum = 0;
	int num_pixel1 = 0;
	for (int j = y1; j < y2; ++j){
		for (int i = x1; i < x2; ++i){
			if(isValidMeasure(depth_map.at<ushort>(i,j))){
				if( depth_map.at<ushort>(i,j) > 500 && depth_map.at<ushort>(i,j) < 7000){
					depth_sum += depth_map.at<ushort>(i,j);
					++num_pixel1;
				}
			}
		}
	}
	ushort dist_mean = 0;
	if(num_pixel1 > 0)
		dist_mean = depth_sum/num_pixel1;
	else
		return 0;

	// obtain foreground
	unsigned long depth_sum1 = 0;
	int num_pixel2 = 0;
	for (int j = y1; j < y2; ++j){
		for (int i = x1; i < x2; ++i){
			if(isValidMeasure(depth_map.at<ushort>(i,j))){
				if( depth_map.at<ushort>(i,j) <= dist_mean && depth_map.at<ushort>(i,j) > 500){
					depth_sum1 += depth_map.at<ushort>(i,j);
					++num_pixel2;
				}
			}
		}
	}
	
	ushort distance = 0;
	if(num_pixel2 > 0)
		distance = depth_sum1/num_pixel2;
	else
		distance = 0;

	if(num_pixel2 >= (int)(0.1 * num_pixel1))  // if foreground pixel > threshold
		return distance;
	else
		return dist_mean;
		
}

// intersectionArea/trackArea
float DTOverlap(const cv::Rect& box1, const cv::Rect& box2)
{
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }
	float colInt =  std::min(box1.x+box1.width,box2.x+box2.width) - std::max(box1.x, box2.x);
	float rowInt =  std::min(box1.y+box1.height,box2.y+box2.height) - std::max(box1.y,box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / area2;
}

/*
float calDispZlj(const cv::Mat &disp, const cv::Rect target_box){
	// clip the bbox, the original code can also be used.
	// 	clip the bbox
	int x1 = std::max(0, target_box.x);
	int y1 = std::max(0, target_box.y);
	int x2 = std::min(target_box.x + target_box.width , disp.cols-1);
	int y2 = std::min(target_box.y + target_box.height, disp.rows-1);
	int cx = (x1 + x2) / 2;
	int cy = (y1 + y2) / 2;
	int w = (x2 - x1) / 2;
	int h = (y2 - y1) / 2;
	assert(cx >= 0 && cy >= 0);
	assert(w >= 0 && h >= 0);

	float sigma = 0.1;
	float weight_sum = 0;
	float Z;
	uchar* data = disp.data;
	int step = disp.step;
	for (int j = y1; j <= y2; ++j){
		for (int i = x1; i <= x2; ++i){
			// x, y, the postion of center of the object
			// z, the value represent the depth of the object
			// calculate in former, then just look up the tabel!
			float weight = 1 / (2 * 3.14159 * pow(sigma, 2)) * exp(-(pow(i - cx, 2) + pow(j - cy, 2)) / (2 * pow(sigma, 2)));
			weight_sum += weight;
			Z += (*(data + j * step + i)) * weight;
		}
	}
	Z = Z / weight_sum;
	int count = 0, gap = 2;
	float cum_Z = 0;
	for (; count < 10; gap++){
		count = 0, cum_Z = 0;
		for (int j = y1; j <= y2; ++j){
			for (int i = x1; i <= x2; ++i){
				if (*(data + j * step + i) > Z - gap
					&& *(data + j * step + i) != 255){
					cum_Z += (*(data + j * step + i));
				count++;
					}
			}
		}
	}

	//    assert(count != 0);
	//    std::cout << "hh " << Z<<std::endl;
	if (count != 0)
		return cum_Z / count;
	else
		return Z;
}

ushort computeDistance(const cv::Mat &depth_map, const cv::Rect target_box)
{
	// 	clip the bbox
	int x1 = std::max(0, target_box.x + (int)(0.25*target_box.width));
	int y1 = std::max(0, target_box.y + (int)(0.25*target_box.height));
	int x2 = std::min(target_box.x + (int)(0.75*target_box.width), depth_map.cols);
	int y2 = std::min(target_box.y + (int)(0.75*target_box.height), depth_map.rows);

	int cx = (x1 + x2) / 2;
	int cy = (y1 + y2) / 2;
	int x1_n = std::max(cx - 30, x1);
	int y1_n = std::max(cy - 30, y1);
	int x2_n = std::min(cx + 30, x2);
	int y2_n = std::min(cy + 30, y2);

	// compute depth mean
	unsigned long depth_sum = 0;
	int num_pixel = 0;
	for (int j = y1_n; j <= y2_n; ++j){
		for (int i = x1_n; i <= x2_n; ++i){
			if(depth_sum > 1000000000){
				return 0;
			}
			ushort pixel_value = depth_map.at<ushort>(i,j);
			if(pixel_value < 5000 && pixel_value > 500){ // filter the depth
				depth_sum += pixel_value;
				++num_pixel;
			}
		}
	}
	if(num_pixel >= 50){
		return depth_sum/num_pixel;
	}
	else{
		return 0;
	}
}
*/