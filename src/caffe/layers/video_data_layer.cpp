#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  this->StopInternalThread();
}



template <typename Dtype>
int VideoDataLayer<Dtype>::Rand(int n) {

    if(n<1)
      {return 0;}
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));


  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_length = this->layer_param_.video_data_param().new_length();
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  string root_folder = this->layer_param_.video_data_param().root_folder();
  const int num_segments = this->layer_param_.video_data_param().num_segments();//lk

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the list file
  const string& source = this->layer_param_.video_data_param().source();
  const bool use_temporal_jitter = this->layer_param_.video_data_param().use_temporal_jitter();
  const bool use_image = this->layer_param_.video_data_param().use_image();
  const int sampling_rate = this->layer_param_.video_data_param().sampling_rate();
  const bool use_multiple_label = this->layer_param_.video_data_param().use_multiple_label();
  if (use_multiple_label) {
    CHECK(this->layer_param_.video_data_param().has_num_of_labels()) <<
    "number of labels must be set together with use multiple labels";

  }
  const int num_of_labels = this->layer_param_.video_data_param().num_of_labels();

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  int count = 0;
  string filename, labels;
  int start_frm, label;

  if (!use_multiple_label) {
    if ((!use_image) && use_temporal_jitter){
      while (infile >> filename >> label) {
        file_list_.push_back(filename);
        label_list_.push_back(label);
        shuffle_index_.push_back(count);
        count++;
      }
    } else {//run here
      while (infile >> filename >> start_frm >> label) {
        file_list_.push_back(filename);
        start_frm_list_.push_back(start_frm);
        label_list_.push_back(label);
        shuffle_index_.push_back(count);
        count++;
  	  }
    }
  } else {
    if ((!use_image) && use_temporal_jitter){
      while (infile >> filename >> labels) {
        file_list_.push_back(filename);
        shuffle_index_.push_back(count);
        vector<int> label_set;
        int tmp_int;
        stringstream sstream(labels);
        while (sstream >> tmp_int) {
          label_set.push_back(tmp_int);
          if (sstream.peek() == ',')
            sstream.ignore();
        }
        multiple_label_list_.push_back(label_set);
        label_list_.push_back(label_set[0]);
        count++;
      }
    } else {
      while (infile >> filename >> start_frm >> labels) {
        file_list_.push_back(filename);
        start_frm_list_.push_back(start_frm);
        shuffle_index_.push_back(count);
        vector<int> label_set;
        int tmp_int;
        stringstream sstream(labels);
        while (sstream >> tmp_int) {
          label_set.push_back(tmp_int);
          if (sstream.peek() == ',')
            sstream.ignore();
        }
        multiple_label_list_.push_back(label_set);
        label_list_.push_back(label_set[0]);
        count++;
      }
    }
  }
  infile.close();

  if (this->layer_param_.video_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleClips();
  }

  if (count==0){
	  LOG(INFO) << "Failed to read the clip list" << std::endl;
  }
  lines_id_ = 0;
  LOG(INFO) << "A total of " << shuffle_index_.size() << " video chunks.";

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(shuffle_index_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read a data point, and use it to initialize the top blob.
  VolumeDatum datum;
  int id = shuffle_index_[lines_id_];
  if (!use_image){
   if (use_temporal_jitter){
    CHECK(ReadVideoToVolumeDatum((root_folder + file_list_[0]).c_str(), 0, label_list_[0],
                               new_length, new_height, new_width, sampling_rate, &datum));
   }
   else
    CHECK(ReadVideoToVolumeDatum((root_folder + file_list_[id]).c_str(), start_frm_list_[id], label_list_[id],
                           	   	   new_length, new_height, new_width, sampling_rate, &datum));
  }
  else{//run here

  // int average_duration ;vector<int> offsets;
  // average_duration = (int) start_frm_list_[id]/num_segments;
  // if(average_duration<16)
  //   {
  //     average_duration=((int) start_frm_list_[id]-16)/num_segments;        
  //       for (int i = 0; i < num_segments; ++i){
  //                int offset = Rand(average_duration);
  //                offsets.push_back(offset+i*average_duration+1);
  //              }
  //   }
  //   else
  //   { 
  //     for (int i = 0; i < num_segments; ++i){
  //               int offset = Rand(average_duration - 15);
  //               offsets.push_back(offset+i*average_duration+1);}
  //   }
        
        vector<int> offsets;  
        // samplerate_lk=Rand(3)+2;
        // if((int) start_frm_list_[id]-samplerate_lk*8<3)   // 8 should be new_length
        //   {samplerate_lk=1;}
        int  average_duration=((int) start_frm_list_[id]-sampling_rate*new_length)/num_segments;        
        for (int i = 0; i < num_segments-1; ++i){
                 int offset = Rand(average_duration);
                 offsets.push_back(offset+i*average_duration+1);
               }
                 int yushu=(int) start_frm_list_[id]%num_segments;
                 int offset = Rand(average_duration+yushu);
                 offsets.push_back(offset+(num_segments-1)*average_duration+1);
                 
     CHECK(ReadImageSequenceToVolumeDatum_Seg((root_folder + file_list_[id]).c_str(), offsets, label_list_[id],
                             new_length, new_height, new_width, sampling_rate, &datum));
  
  

   //CHECK(ReadImageSequenceToVolumeDatum((root_folder + file_list_[id]).c_str(), start_frm_list_[id], label_list_[id],
     //                         new_length, new_height, new_width, sampling_rate, &datum));
  }

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size_LK: " << top[0]->shape(0) << ","
        << top[0]->shape(1) << "," << top[0]->shape(2) << ","
        << top[0]->shape(3) << "," << top[0]->shape(4);

  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  if (use_multiple_label)
    label_shape.push_back(num_of_labels);
  top[1]->Reshape(label_shape);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleClips() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(shuffle_index_.begin(), shuffle_index_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_length = video_data_param.new_length();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  string root_folder = video_data_param.root_folder();
  const bool use_image = video_data_param.use_image();
  const bool use_temporal_jitter = video_data_param.use_temporal_jitter();
  int sampling_rate = video_data_param.sampling_rate();
  const int max_sampling_rate = video_data_param.max_sampling_rate();
  const bool use_sampling_rate_jitter = video_data_param.use_sampling_rate_jitter();
  const bool show_data = video_data_param.show_data();
  const int num_segments = video_data_param.num_segments();//lk

  const bool use_multiple_label = this->layer_param_.video_data_param().use_multiple_label();
  if (use_multiple_label) {
    CHECK(this->layer_param_.video_data_param().has_num_of_labels()) <<
    "number of labels must be set together with use multiple labels";

  }
  const int num_of_labels = this->layer_param_.video_data_param().num_of_labels();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  // Read a data point, and use it to initialize the top blob.
  VolumeDatum datum;
  int id = shuffle_index_[lines_id_];
  if (!use_image){
    if (use_temporal_jitter){
      ReadVideoToVolumeDatum((root_folder + file_list_[0]).c_str(), 0, label_list_[0],
            new_length, new_height, new_width, sampling_rate, &datum);
    } else {
      ReadVideoToVolumeDatum((root_folder + file_list_[id]).c_str(), start_frm_list_[id], label_list_[id],
            new_length, new_height, new_width, sampling_rate, &datum);
    }
  } else {
   //run here

        vector<int> offsets;    
        // samplerate_lk=Rand(3)+2;
        // if((int) start_frm_list_[id]-samplerate_lk*8<0)
        //   {samplerate_lk=2;}
        int  average_duration=((int) start_frm_list_[id]-sampling_rate*new_length)/num_segments;        
        for (int i = 0; i < num_segments-1; ++i){
                 int offset = Rand(average_duration);
                 offsets.push_back(offset+i*average_duration+1);
               }
                 int yushu=(int) start_frm_list_[id]%num_segments;
                 int offset = Rand(average_duration+yushu);
                 offsets.push_back(offset+(num_segments-1)*average_duration+1);
                 
     CHECK(ReadImageSequenceToVolumeDatum_Seg((root_folder + file_list_[id]).c_str(), offsets, label_list_[id],
                             new_length, new_height, new_width, sampling_rate, &datum));
  
  //ori 
   //CHECK(ReadImageSequenceToVolumeDatum((root_folder + file_list_[id]).c_str(), start_frm_list_[id], label_list_[id],
     //                         new_length, new_height, new_width, sampling_rate, &datum));
  }

  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int dataset_size = shuffle_index_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    if (use_sampling_rate_jitter) {
      sampling_rate = caffe::caffe_rng_rand() % (max_sampling_rate) + 1;
    }
    timer.Start();
    CHECK_GT(dataset_size, lines_id_);
    bool read_status;
    int id = this->shuffle_index_[this->lines_id_];
    // if (!use_image){
    // 	if (!use_temporal_jitter){
    //         read_status = ReadVideoToVolumeDatum((root_folder + this->file_list_[id]).c_str(), this->start_frm_list_[id],
    //         		this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
    //     }else{
    //     	read_status = ReadVideoToVolumeDatum((root_folder + this->file_list_[id]).c_str(), -1,
    //     			this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
    //     }
    // }
     //else {
        if (!use_temporal_jitter) {// run here
        
        vector<int> offsets;
        // sampling_rate=Rand(3)+2;
        // if((int) start_frm_list_[id]-sampling_rate*8<0)
        //   {sampling_rate=2;}
        int  average_duration=((int) start_frm_list_[id]-sampling_rate*new_length)/num_segments;        
        for (int i = 0; i < num_segments-1; ++i){
                 int offset = Rand(average_duration);
                 offsets.push_back(offset+i*average_duration+1);
               }
               int yushu=(int) start_frm_list_[id]%num_segments;
                 int offset = Rand(average_duration+yushu);
                 offsets.push_back(offset+(num_segments-1)*average_duration+1);

  

     read_status =ReadImageSequenceToVolumeDatum_Seg((root_folder + file_list_[id]).c_str(), offsets, label_list_[id],
                             new_length, new_height, new_width, sampling_rate, &datum);

        } 
        // else {
        // 	int num_of_frames = this->start_frm_list_[id];
        // 	int use_start_frame;
        // 	if (num_of_frames < new_length * sampling_rate){
        // 	    LOG(INFO) << "not enough frames; having " << num_of_frames;
        // 	    read_status = false;
        // 	} else {
        // 	    if (this->phase_ == TRAIN)
        // 	    	use_start_frame = caffe_rng_rand()%(num_of_frames-new_length*sampling_rate+1) + 1;
        // 	    else
        // 	    	use_start_frame = 0;
        // 	    read_status = ReadImageSequenceToVolumeDatum((root_folder + this->file_list_[id]).c_str(), use_start_frame,
        // 	    			    this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
        // 	}
        // }
   // }

    if (this->phase_ == TEST){
        CHECK(read_status) << "Testing must not miss any example";
    }

    if (!read_status) {
        this->lines_id_++;
        if (this->lines_id_ >= dataset_size) {
        	// We have reached the end. Restart from the first.
        	DLOG(INFO) << "Restarting data prefetching from start.";
        	this->lines_id_ = 0;
        	if (this->layer_param_.video_data_param().shuffle()){
        		ShuffleClips();
        	}
        }
        item_id--;
        continue;
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply transformations (mirror, crop...) to the video
    vector<int> shape_vec(5, 0);
    shape_vec[0] = item_id;
    int offset = batch->data_.offset(shape_vec);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
   	this->data_transformer_->VideoTransform(datum, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    if (!use_multiple_label) {
      prefetch_label[item_id] = datum.label();
    } else {
      caffe_set<Dtype>(num_of_labels, Dtype(0), prefetch_label + item_id * num_of_labels);
      for (int index= 0; index < this->multiple_label_list_[id].size(); index++) {
          prefetch_label[item_id * num_of_labels +
            this->multiple_label_list_[id][index]] = Dtype(1);
        }
    }

    // Show visualizationVideoTransform
    if (show_data){
    	const Dtype* data_buffer = (Dtype*)(prefetch_data + offset);
      LOG(ERROR) << "starting show_data.";
        int image_size, channel_size;
       	image_size = top_shape[3] * top_shape[4];
        channel_size = top_shape[2] * image_size;
        for (int l = 0; l < top_shape[2]; ++l) {
        	for (int c = 0; c < top_shape[1]; ++c) {
        		cv::Mat img;
        		char ch_name[64];
        		BufferToGrayImage(data_buffer + c * channel_size + l * image_size, top_shape[3], top_shape[4], &img);
        		sprintf(ch_name, "./Channel_%d_%d.jpg", l,c);
        		// cv::namedWindow(ch_name, CV_WINDOW_AUTOSIZE);
        		// cv::imshow(ch_name, img);
            cv::imwrite(ch_name, img);
        	}
        	cv::waitKey(10000);
        }
    }

    // go to the next iter
    this->lines_id_++;
    if (lines_id_ >= dataset_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleClips();
      }
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

  // std::ofstream out("profile_inference.log", std::ofstream::out | std::ofstream::app);
  // out << "data CPU " << batch_timer.MilliSeconds() << "\n";
  // out.close();
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
#endif  // USE_OPENCV
