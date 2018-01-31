/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


namespace caffe {


void ImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
	for (int c = 0; c < 3; ++c) {
	  for (int h = 0; h < img->rows; ++h) {
		for (int w = 0; w < img->cols; ++w) {
			buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	  }
	}
}
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c){
    int idx = 0;
	for (int h = 0; h < img->rows; ++h) {
	    for (int w = 0; w < img->cols; ++w) {
		    buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	}
}

void GrayImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
    for (int h = 0; h < img->rows; ++h) {
	  for (int w = 0; w < img->cols; ++w) {
		buffer[idx++] = img->at<unsigned char>(h, w);
	  }
	}
}
void BufferToGrayImage(const char* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = buffer[idx++];
	  }
	}
}
void BufferToGrayImage(const float* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = (unsigned char)(buffer[idx++]);
	  }
	}
}
void BufferToGrayImage(const double* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = (unsigned char)(buffer[idx++]);
	  }
	}
}

void BufferToColorImage(const char* buffer, const int height, const int width, cv::Mat* img){
	img->create(height, width, CV_8UC3);
	for (int c=0; c<3; c++) {
		for (int h = 0; h < height; ++h) {
		  for (int w = 0; w < width; ++w) {
			img->at<cv::Vec3b>(h, w)[c] = buffer[c * width * height + h * width + w];
		  }
		}
	}
}

bool ReadVideoToVolumeDatum(const char* filename, const int start_frm,
  const int label, const int length, const int height, const int width,
  const int sampling_rate, VolumeDatum* datum){
  if (!ReadVideoToVolumeDatumHelper(filename, start_frm, label, length,
      height, width, sampling_rate, datum)) {
      return ReadVideoToVolumeDatumHelperSafe(filename, start_frm, label,
        length, height, width, sampling_rate, datum);
  } else
      return true;
}

bool ReadVideoToVolumeDatumHelper(const char* filename, const int start_frm,
  const int label, const int length, const int height, const int width,
  const int sampling_rate, VolumeDatum* datum){
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  char *buffer = NULL;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;
  int use_start_frm = start_frm;

  cap.open(filename);
  if (!cap.isOpened()){
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  datum->set_channels(3);
  datum->set_length(length);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames<length*sampling_rate){
    LOG(INFO) << filename << " does not have enough frames; having "
    << num_of_frames;
    return false;
  }

  // If start_frame == -1, then use random temporal jitering
  if (start_frm < 0){
    use_start_frm = caffe_rng_rand()%(num_of_frames-length*sampling_rate+1);
  }

  offset = 0;
  CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

  if (use_start_frm)
    cap.set(CV_CAP_PROP_POS_FRAMES, use_start_frm);

  int end_frm = use_start_frm + length * sampling_rate;
  CHECK_LE(end_frm, num_of_frames)
    << "end frame must be less or equal to num of frames";

  for (int i=use_start_frm; i<end_frm; i+=sampling_rate){
		if (sampling_rate > 1)
			cap.set(CV_CAP_PROP_POS_FRAMES, i);
		if (height > 0 && width > 0){
			cap.read(img_origin);
			if (!img_origin.data){
				LOG(INFO) << filename << " has no data at frame " << i;
				if (buffer!=NULL)
					delete[] buffer;
				return false;
			}
			cv::resize(img_origin, img, cv::Size(width, height));
		}
		else
			cap.read(img);
		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << filename;
			if (buffer!=NULL)
				delete[] buffer;
			return false;
		}

		if (i==use_start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
	cap.release();
 	return true;
}

bool ReadVideoToVolumeDatumHelperSafe(const char* filename, const int start_frm,
  const int label, const int length, const int height, const int width,
  const int sampling_rate, VolumeDatum* datum){
  cv::VideoCapture cap;
	cv::Mat img, img_origin;
  	char *buffer = NULL;
	int offset = 0;
	int channel_size = 0;
	int image_size = 0;
	int data_size = 0;
	int use_start_frm = start_frm;

	cap.open(filename);
	if (!cap.isOpened()){
		LOG(ERROR) << "Cannot open " << filename;
		return false;
	}

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	if (num_of_frames<length*sampling_rate){
		LOG(INFO) << filename << " does not have enough frames; having "
    << num_of_frames;
		return false;
	}
	if (start_frm < 0){
		use_start_frm = caffe_rng_rand()%(num_of_frames-length*sampling_rate+1);
	}

	offset = 0;
	CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

  // Instead of random acess, do sequentically access (avoid key-frame issue)
	// This will keep use_start_frm frames
	int sequential_counter = 0;
	while (sequential_counter < use_start_frm) {
		cap.read(img_origin);
		sequential_counter++;
	}

	int end_frm = use_start_frm + length * sampling_rate;
	CHECK_LE(end_frm, num_of_frames) << "end frame must be less or equal to num of frames";

	for (int i=use_start_frm; i<end_frm; i++){
		if (sampling_rate > 1) {
			// If sampling_rate > 1, purposely keep some frames
			if ((i-use_start_frm) % sampling_rate !=0) {
				cap.read(img_origin);
				continue;
			}
		}
		if (height > 0 && width > 0){
			cap.read(img_origin);
			if (!img_origin.data){
				LOG(INFO) << filename << " has no data at frame " << i;
				if (buffer!=NULL)
					delete[] buffer;
				return false;
			}
			cv::resize(img_origin, img, cv::Size(width, height));
		}
		else
			cap.read(img);
		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << filename;
			if (buffer!=NULL)
				delete[] buffer;
			return false;
		}

		if (i==use_start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
	cap.release();
	return true;
}

// flow_x read the 3 channel optical flow

bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	cv::Mat img, img_origin;
	char *buffer = NULL;
	int offset = 0;
	int channel_size = 0;
	int image_size = 0;
	int data_size = 0;

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0;
	int end_frm = start_frm + length * sampling_rate;
	for (int i=start_frm; i<end_frm; i+=sampling_rate){
	
		sprintf(fn_im, "%s/image_%06d.jpg", img_dir, i); //this is original code,not TSN
	
		if (height > 0 && width > 0) {
		    img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
		    if (!img_origin.data) {
			LOG(ERROR) << "Could not open or find file " << fn_im;
		    	return false;
		    }
		    cv::resize(img_origin, img, cv::Size(width, height));
		    img_origin.release();
		} else {
		  img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
		}

		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << fn_im;
			return false;
		}

		if (i==start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}


bool ReadImageSequenceToVolumeDatum_Seg(const char* img_dir, const vector<int> offsets, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	cv::Mat img, img_origin;
	char *buffer = NULL;
	int offset = 0;
	int channel_size = 0;
	int image_size = 0;
	int data_size = 0;

	datum->set_channels(3*offsets.size());
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0; 

	for (int offsets_i= 0; offsets_i < offsets.size(); ++offsets_i){
		int start_frm = offsets[offsets_i];
		//LOG(ERROR) <<"  img_dirv " <<img_dir<<"  start_frm " << start_frm<<" offsets_i "<<offsets_i;
		//LOG(ERROR) <<"  start_frm: " << start_frm<<"  sampling_rate:" << sampling_rate<<" offsets_i: "<<offsets_i;
		int end_frm = start_frm + length * sampling_rate;
		for (int i=start_frm; i<end_frm; i+=sampling_rate){
			//sprintf(fn_im, "%s/%06d.jpg", img_dir, i);
			//int frame_order=0;
			sprintf(fn_im, "%s/image_%04d.jpg", img_dir, i); 
			//sprintf(fn_im, "%s/flow_x_%04d.jpg", img_dir, i); 
			if (height > 0 && width > 0) {
			    img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
			    if (!img_origin.data) {
				LOG(ERROR) << "Could not open or find file " << fn_im;
				return false;
			    }
			    cv::resize(img_origin, img, cv::Size(width, height));
			    img_origin.release();
			} else {
			  img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
			}

			if (!img.data){
				LOG(ERROR) << "Could not open or find file " << fn_im;
				return false;
			}

			if (i==start_frm && offsets_i==0){
				datum->set_height(img.rows);
				datum->set_width(img.cols);
				image_size = img.rows * img.cols;
				// channel_size = image_size * length*offsets.size();
				// data_size = channel_size * 3;

				channel_size = image_size * length*3;
			    data_size = channel_size * offsets.size();

				buffer = new char[data_size];
			}
			
			for (int c=0; c<3; c++){
				
			  	ImageChannelToBuffer(&img, buffer + c * image_size * length + offsets_i*channel_size+offset, c);
			 
			  //ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c); original
			}

		offset += image_size;
		}
		offset = 0;
	}
	

	//CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}

/* //This is for end0614
bool ReadImageSequenceToVolumeDatum_Seg(const char* img_dir, const vector<int> offsets, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	cv::Mat img, img_origin;
	char *buffer = NULL;
	int offset = 0;
	int channel_size = 0;
	int image_size = 0;
	int data_size = 0;

	datum->set_channels(3*offsets.size());
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0; int all_frame_order=0;

	for (int offsets_i= 0; offsets_i < offsets.size(); ++offsets_i){
		int start_frm = offsets[offsets_i];
		//LOG(ERROR) <<"  img_dirv " <<img_dir<<"  start_frm " << start_frm<<" offsets_i "<<offsets_i;
		int end_frm = start_frm + length * sampling_rate;
		for (int i=start_frm; i<end_frm; i+=sampling_rate){
			//sprintf(fn_im, "%s/%06d.jpg", img_dir, i);
			//int frame_order=0;
			sprintf(fn_im, "%s/image_%04d.jpg", img_dir, i); 
			//sprintf(fn_im, "%s/flow_x_%04d.jpg", img_dir, i); 
			if (height > 0 && width > 0) {
			    img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
			    if (!img_origin.data) {
				LOG(ERROR) << "Could not open or find file " << fn_im;
				return false;
			    }
			    cv::resize(img_origin, img, cv::Size(width, height));
			    img_origin.release();
			} else {
			  img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
			}

			if (!img.data){
				LOG(ERROR) << "Could not open or find file " << fn_im;
				return false;
			}

			if (i==start_frm && offsets_i==0){
				datum->set_height(img.rows);
				datum->set_width(img.cols);
				image_size = img.rows * img.cols;
				//channel_size = image_size * length;
				channel_size = image_size * length*offsets.size();
				data_size = channel_size * 3;
				buffer = new char[data_size];
			}
			//for (int offsets_i= 0; offsets_i < offsets.size(); ++offsets_i){
			for (int c=0; c<3; c++){
				//ImageChannelToBuffer(&img, buffer + c *offsets_i* image_size * length + offset, c);
			  	ImageChannelToBuffer(&img, buffer + c * image_size * length + (all_frame_order%3)*channel_size+offset, c);
			  //ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c); original
			}

			//frame_order=frame_order+1;
			all_frame_order=all_frame_order+1;
			if(0==(all_frame_order%3))
				{offset += image_size;}
		}
		//offset = 0;
	}
	

	//CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}*/
/*
bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	char fn_im_y[256];
	char fn_im_gray[256];
	cv::Mat img,img_y,img_gray;
	char *buffer = NULL;
	int offset = 0;
	int channel_size = 0;
	int image_size = 0;
	int data_size = 0;

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0;
	int end_frm = start_frm + length * sampling_rate;
	for (int i=start_frm; i<end_frm; i+=sampling_rate){
	   	sprintf(fn_im, "%s/flow_x_%04d.jpg", img_dir, i); 
		sprintf(fn_im_y, "%s/flow_y_%04d.jpg", img_dir, i); 		
		sprintf(fn_im_gray, "%s/image_%04d.jpg", img_dir, i); 		
			
		cv::Mat cv_img_origin_x = cv::imread(fn_im, CV_LOAD_IMAGE_GRAYSCALE);		
		cv::Mat cv_img_origin_y = cv::imread(fn_im_y, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat cv_img_origin_gray = cv::imread(fn_im_gray, CV_LOAD_IMAGE_GRAYSCALE);
		
		if (!cv_img_origin_x.data || !cv_img_origin_y.data || !cv_img_origin_gray.data){
			LOG(ERROR) << "Could not load file " << fn_im << " or " << fn_im_y<< " or " << fn_im_gray;
			return false;
		}
		
		if (height > 0 && width > 0){
			cv::resize(cv_img_origin_x, img, cv::Size(width, height));
			cv::resize(cv_img_origin_y, img_y, cv::Size(width, height));
			cv::resize(cv_img_origin_gray, img_gray, cv::Size(width, height));
		}else{
			img = cv_img_origin_x;
			img_y = cv_img_origin_y;
			img_gray=cv_img_origin_gray;
		}

		if (i==start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length*3;
			data_size = channel_size * 1;
			buffer = new char[data_size];
		}
		
		GrayImageToBuffer(&img, buffer  + offset);
		GrayImageToBuffer(&img_y, buffer  + offset);		
		GrayImageToBuffer(&img_gray, buffer  + offset);
		offset += image_size*3;
	}

	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}
*/

// optical flow 3(x+y),6channels
/*
bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	char fn_im_y[256];
	char fn_im1[256];
	char fn_im_y1[256];
	char fn_im2[256];
	char fn_im_y2[256];
	
	cv::Mat img,img_y,img1,img_y1,img2,img_y2;
	char *buffer = NULL;
	int offset = 0;
	int channel_size = 0;
	int image_size = 0;
	int data_size = 0;

	datum->set_channels(6);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0;
	int end_frm = start_frm + length * sampling_rate;
	for (int i=start_frm; i<end_frm; i+=sampling_rate){
	   	
	   	sprintf(fn_im, "%s/flow_x_%04d.jpg", img_dir, i); 
		sprintf(fn_im_y, "%s/flow_y_%04d.jpg", img_dir, i); 		
		sprintf(fn_im1, "%s/flow_x_%04d.jpg", img_dir, i+1); 
		sprintf(fn_im_y1, "%s/flow_y_%04d.jpg", img_dir, i+1); 		
		sprintf(fn_im2, "%s/flow_x_%04d.jpg", img_dir, i+2); 
		sprintf(fn_im_y2, "%s/flow_y_%04d.jpg", img_dir, i+2); 		
		
			
		cv::Mat cv_img_origin_x = cv::imread(fn_im, CV_LOAD_IMAGE_GRAYSCALE);		
		cv::Mat cv_img_origin_y = cv::imread(fn_im_y, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat cv_img_origin_x1 = cv::imread(fn_im1, CV_LOAD_IMAGE_GRAYSCALE);		
		cv::Mat cv_img_origin_y1 = cv::imread(fn_im_y1, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat cv_img_origin_x2 = cv::imread(fn_im2, CV_LOAD_IMAGE_GRAYSCALE);		
		cv::Mat cv_img_origin_y2 = cv::imread(fn_im_y2, CV_LOAD_IMAGE_GRAYSCALE);
		
		if (!cv_img_origin_x.data || !cv_img_origin_y.data || !cv_img_origin_x1.data || !cv_img_origin_y1.data||!cv_img_origin_x2.data||!cv_img_origin_x2.data){
			LOG(ERROR) << "Could not load file " << fn_im << " or " << fn_im_y<< " or " << fn_im1;
			LOG(ERROR) << "Could not load file " << fn_im_y1 << " or " << fn_im2<< " or " << fn_im_y2;
			return false;
		}
		
		if (height > 0 && width > 0){
			cv::resize(cv_img_origin_x, img, cv::Size(width, height));
			cv::resize(cv_img_origin_y, img_y, cv::Size(width, height));
			cv::resize(cv_img_origin_x1, img1, cv::Size(width, height));
			cv::resize(cv_img_origin_y1, img_y1, cv::Size(width, height));
			cv::resize(cv_img_origin_x2, img2, cv::Size(width, height));
			cv::resize(cv_img_origin_y2, img_y2, cv::Size(width, height));
		}else{
			img = cv_img_origin_x;
			img_y = cv_img_origin_y;
			
						img1 = cv_img_origin_x1;
			img_y1 = cv_img_origin_y1;
						img2 = cv_img_origin_x2;
			img_y2 = cv_img_origin_y2;
		}

		if (i==start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length*6;
			data_size = channel_size * 1;
			buffer = new char[data_size];
		}
		
		GrayImageToBuffer(&img, buffer  + offset);
		GrayImageToBuffer(&img_y, buffer  + offset);		
		GrayImageToBuffer(&img1, buffer  + offset);
		GrayImageToBuffer(&img_y1, buffer  + offset);	
		GrayImageToBuffer(&img2, buffer  + offset);
		GrayImageToBuffer(&img_y2, buffer  + offset);			
		
		offset += image_size*6;
	}

	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}
*/
template <>
bool load_blob_from_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;

	blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;

	blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_uint8_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;
	blob->Reshape(shape);

	buff = blob->mutable_cpu_data();

	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (float)temp_buff[i];

	delete []temp_buff;
	return true;
}

template <>
bool load_blob_from_uint8_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;
	blob->Reshape(shape);

	buff = blob->mutable_cpu_data();


	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (double)temp_buff[i];

	delete []temp_buff;
	return true;
}


template <>
bool save_blob_to_binary<float>(Blob<float>* blob, const string fn_blob, int num_index){
	FILE *f;
	float *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	c = blob->shape(1);
	if (blob->num_axes() > 2)
		l = blob->shape(2);
	else
		l = 1;
	if (blob->num_axes() > 3)
		h = blob->shape(3);
	else
		h = 1;
	if (blob->num_axes() > 4)
		w = blob->shape(4);
	else
		w = 1;

	if (num_index<0){
		n = blob->shape(0);
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
		buff = blob->mutable_cpu_data() + num_index * c * l * h * w;
	}

	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool save_blob_to_binary<double>(Blob<double>* blob, const string fn_blob, int num_index){
	FILE *f;
	double *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	c = blob->shape(1);
	if (blob->num_axes() > 2)
		l = blob->shape(2);
	else
		l = 1;
	if (blob->num_axes() > 3)
		h = blob->shape(3);
	else
		h = 1;
	if (blob->num_axes() > 4)
		w = blob->shape(4);
	else
		w = 1;
	if (num_index<0){
		n = blob->shape(0);
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
		buff = blob->mutable_cpu_data() + num_index * c * l * h * w;
	}


	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}


}
