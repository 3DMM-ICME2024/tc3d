#ifndef CAFFE_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_HPP_ 
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {
    template <typename Dtype>
    class SoftmaxWithCrossEntropyLossLayer : public LossLayer<Dtype> {
     public:
      /**
       * @param param provides LossParameter loss_param, with options:
       *  - normalize (optional, default true)
       *    If true, the loss is normalized by the number of (nonignored) labels
       *    present; otherwise the loss is simply summed over spatial locations.
       */
      explicit SoftmaxWithCrossEntropyLossLayer(const LayerParameter& param)
          : LossLayer<Dtype>(param) {}
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

      virtual inline const char* type() const { return "SoftmaxWithCrossEntropyLoss"; }

     protected:
      /// @copydoc SoftmaxWithCrossEntropyLossLayer
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
      /**
       * @brief Computes the softmax loss error gradient w.r.t. the predictions.
       *
       * Gradients cannot be computed with respect to the target inputs (bottom[1]),
       * so this method ignores bottom[1] and requires !propagate_down[1], crashing
       * if propagate_down[1] is set.
       *
       * @param top output Blob vector (length 1), providing the error gradient with
       *      respect to the outputs
       *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
       *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
       *      as @f$ \lambda @f$ is the coefficient of this layer's output
       *      @f$\ell_i@f$ in the overall Net loss
       *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
       *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
       *      (*Assuming that this top Blob is not used as a bottom (input) by any
       *      other layer of the Net.)
       * @param propagate_down see Layer::Backward.
       *      propagate_down[1] must be false as we can't compute gradients with
       *      respect to the targets.
       * @param bottom input Blob vector (length 2)
       *   -# @f$ (N \times C \times H \times W) @f$
       *      the predictions @f$ x @f$; Backward computes diff
       *      @f$ \frac{\partial E}{\partial x} @f$
       *   -# @f$ (N \times C \times H \times W) @f$
       *      the targets -- ignored as we can't compute their error gradients
       */
      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


      /// The internal SoftmaxLayer used to map predictions to a distribution.
      shared_ptr<Layer<Dtype> > softmax_layer_;
      /// prob stores the output probability predictions from the SoftmaxLayer.
      Blob<Dtype> predict_prob_;
      Blob<Dtype> target_prob_;
      /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
      vector<Blob<Dtype>*> softmax_predict_bottom_vec_;
      vector<Blob<Dtype>*> softmax_target_bottom_vec_;
      /// top vector holder used in call to the underlying SoftmaxLayer::Forward
      vector<Blob<Dtype>*> softmax_predict_top_vec_;
      vector<Blob<Dtype>*> softmax_target_top_vec_;
      /// Whether to normalize the loss by the total number of values present
      /// (otherwise just by the batch size).
      bool normalize_;

      int softmax_axis_, outer_num_, inner_num_;
    };
}
#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_