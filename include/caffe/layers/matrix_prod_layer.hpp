#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"

namespace caffe {
	
		template <typename Dtype>
		class MatrixProdLayer : public Layer<Dtype> {
		 public:
		  explicit MatrixProdLayer(const LayerParameter& param)
		      : Layer<Dtype>(param) {}
		  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
		  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
		
		  virtual inline const char* type() const { return "MatrixProd"; }
		  virtual inline int ExactNumBottomBlobs() const { return 2; }
		  virtual inline int ExactNumTopBlobs() const { return 1; }
		
		 protected:
		  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
		  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
		  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		
		  int M_; // Batch Size
		  int K_; // Embedding Size
		  int N_; // Dictionary Size
		};
}  // namespace caffe
