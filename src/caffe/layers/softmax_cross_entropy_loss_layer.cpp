#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_predict_bottom_vec_.clear();


  softmax_predict_bottom_vec_.push_back(bottom[0]);
  softmax_target_bottom_vec_.clear();
  softmax_target_bottom_vec_.push_back(bottom[1]);
  softmax_predict_top_vec_.clear();
  softmax_predict_top_vec_.push_back(&predict_prob_);
  softmax_target_top_vec_.clear();
  softmax_target_top_vec_.push_back(&target_prob_);
  softmax_layer_->SetUp(softmax_predict_bottom_vec_, softmax_predict_top_vec_);

  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_predict_bottom_vec_, softmax_predict_top_vec_);
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
      << "bottom[0] and bottom[1] should have the same shape.";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
          << "bottom[0] and bottom[1] should have the same shape.";
  }
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.


  softmax_layer_->Forward(softmax_predict_bottom_vec_, softmax_predict_top_vec_);
  softmax_layer_->Forward(softmax_target_bottom_vec_, softmax_target_top_vec_);
  const Dtype* predict_prob_data = predict_prob_.cpu_data();
  const Dtype* target_prob_data = target_prob_.cpu_data();
  int dim = predict_prob_.count() / outer_num_;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      for (int k = 0; k < predict_prob_.shape(softmax_axis_); ++k) {
        loss -= target_prob_data[i * dim + k * inner_num_ + j]
            * log(std::max(predict_prob_data[i * dim + k * inner_num_ + j], Dtype(FLT_MIN)));
      }
    }
  }
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      for (int k = 0; k < predict_prob_.shape(softmax_axis_); ++k) {
        loss += target_prob_data[i * dim + k * inner_num_ + j]
            * log(std::max(target_prob_data[i * dim + k * inner_num_ + j], Dtype(FLT_MIN)));
      }
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / (outer_num_ * inner_num_);
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(predict_prob_);
  }
}

template <typename Dtype>
void SoftmaxWithCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to target score inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* predict_prob_data = predict_prob_.cpu_data();
    const Dtype* target_prob_data = target_prob_.cpu_data();
    // int dim = predict_prob_.count() / outer_num_;
    caffe_sub(predict_prob_.count(), predict_prob_data, target_prob_data, bottom_diff);
    // for (int i = 0; i < outer_num_; ++i) {
    //   for (int j = 0; j < inner_num_; ++j) {
    //     Dtype sum = 0;
    //     for (int k = 0; k < bottom[0]->shape(softmax_axis_); ++k) {
    //         sum += target_prob_data[i * dim + k * inner_num_ + j];
    //     }
    //     for (int k = 0; k < bottom[0]->shape(softmax_axis_); ++k) {
    //       bottom_diff[i * dim + k * inner_num_ + j] =
    //           sum * predict_prob_data[i * dim + k * inner_num_ + j]
    //           - target_prob_data[i * dim + k * inner_num_ + j];
    //     }
    //   }
    // }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(predict_prob_.count(), loss_weight / (outer_num_ * inner_num_), bottom_diff);
    } else {
      caffe_scal(predict_prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithCrossEntropyLoss);

}  // namespace caffe
