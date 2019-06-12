#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/contrib/makefile/downloads/absl/absl/strings/string_view.h"
#include "time.h"

using namespace cv;
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
void resize_image(const cv::Mat& image,
                             cv::Mat& resized_image){
int height = image.rows;
int width = image.cols;
int resize_h =155;
int resize_w = 220;
cv::resize(image, resized_image, cv::Size(resize_w, resize_h));  
}

tensorflow::Tensor cv_mat_to_tensor(const cv::Mat& image){
  int height = image.rows;
  int width = image.cols;
  int depth = 1;
  tensorflow::Tensor res_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, 1}));

  cv::Mat image2;
  image.convertTo(image2, CV_32FC1);
  //we assume that the image is unsigned char dtype
  const float *source_data = (float*)(image2.data); 

  auto tensor_mapped = res_tensor.tensor<float, 4>();
  for (int y = 0; y < height; ++y) {
    const float* source_row = source_data + (y * width * depth);
    for (int x = 0; x < width; ++x) {
      const float* source_pixel = source_row + (x * depth);
      float b = *(source_pixel);
      //float g = *(source_pixel + 1);
      //float r = *(source_pixel + 2);
      tensor_mapped(0, y, x, 0) = b; //r
      //tensor_mapped(0, y, x, 1) = g;
      //tensor_mapped(0, y, x, 2) = b;
    }
  }
  return res_tensor;
}
int main(int argc, char** argv)
{
  clock_t startTime,endTime;
  startTime=clock();


  Session* session;
  Status status = NewSession(SessionOptions(), &session);//创建新会话Session


  string model_path="test.pb";
  GraphDef graphdef; //Graph Definition for current model


  Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
  if (!status_load.ok()) {
      std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
      std::cout << status_load.ToString() << "\n";
      return -1;
  }
  Status status_create = session->Create(graphdef); //将模型导入会话Session中;
  if (!status_create.ok()) {
      std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
      return -1;
  }
  cout << "Session successfully created."<< endl;
string image_path1 = argv[1];
string image_path2 = argv[2];
string image_filename1=image_path1;
string image_filename2=image_path2;
cv::Mat image1 = cv::imread(image_filename1);
cv::Mat dst1;
cv::cvtColor(image1,dst1,CV_BGR2GRAY);
dst1=dst1/30.564154;
if(!image1.data)                              // Check for invalid input
{
std::cout <<  "Could not open or find the image "  <<image_path1<< std::endl ;    
}
cv::Mat image2 = cv::imread(image_filename2);
cv::Mat dst2;
cv::cvtColor(image2,dst2,CV_BGR2GRAY);
dst2=dst2/30.564154;
if(!image2.data)                              // Check for invalid input
{
std::cout <<  "Could not open or find the image "  <<image_path2<< std::endl ;    
}
//std::cout<< image1<<std::endl;
cv::Mat resized_image1;
resize_image(dst1, resized_image1);
std::cout<<resized_image1.rows<<" "<<resized_image1.cols<<std::endl;
cv::Mat resized_image2;
resize_image(dst2, resized_image2);
std::cout<<resized_image2.rows<<" "<<resized_image2.cols<<std::endl;
//cout<<resized_image1<<endl;
tensorflow::Tensor tmptensor1= cv_mat_to_tensor(resized_image1);
tensorflow::Tensor tmptensor2= cv_mat_to_tensor(resized_image2);
const Tensor& resized_tensor1 = tmptensor1;
  
  //const Tensor& resized_tensor2 = resized_tensors1[1];
  cout<<"resized_tensor1"<<endl;
  std::cout << resized_tensor1.DebugString()<<endl;
  const Tensor& resized_tensor2 = tmptensor2;
  cout<<"resized_tensor2"<<endl;
  std::cout << resized_tensor2.DebugString()<<endl;
  auto resize1 = resized_tensor1.tensor<float, 4>();
  auto resize2 = resized_tensor2.tensor<float, 4>();
  cout<<"resized1"<<endl;
 // for(int m=0;m<155;m++)
 // {
 //   for(int n=0;n<220;n++)
  //  {  if((resize1(0,m,n,0)<0.5)&&(resize1(0,m,n,0)<0.5))
 //     {
 //       cout<<resize1(0,m,n,0)<<",";
  //      cout<<resize2(0,m,n,0)<<",";
 //     }
 //   }
 // }
  cout<<"resize2"<<endl;


  vector<tensorflow::Tensor> outputs1;
  vector<tensorflow::Tensor> outputs2;
  //string output_node = "softmax";
  //string output_node = "concatenate_1/concat:0";
  string output_node =  "lambda_1/Sqrt:0";
  //cout<<output_node<<endl;
  string output_node1 = "sequential_1/dense_2/Relu:0";
  string output_node2 = "sequential_1_1/dense_2/Relu:0";
  //string output_node = "dense_2/BiasAdd:0";
  //Status status_run = session->Run({{"inputs", resized_tensor}}, {output_node}, {}, &outputs);
  //Tensor("input_1:0", shape=(?, 155, 220, 1), dtype=float32)

  Status status_run1 = session->Run({{"input_1:0", resized_tensor1},{"input_2:0", resized_tensor2}}, {output_node}, {}, &outputs1);
  Status status_run2 = session->Run({{"input_2:0", resized_tensor2}}, {output_node2}, {}, &outputs2);
  cout<<"here"<<endl;
  if (!status_run1.ok()) {
      std::cout << "ERROR: RUN1 failed..."  << std::endl;
      std::cout << status_run1.ToString() << "\n";
      return -1;
  }
  if (!status_run2.ok()) {
      std::cout << "ERROR: RUN2 failed..."  << std::endl;
      std::cout << status_run2.ToString() << "\n";
      return -1;
  }
  //Fetch output value
  std::cout << "Output tensor1 size:" << outputs1.size() << std::endl;
  for (std::size_t i = 0; i < outputs1.size(); i++) {
      std::cout << outputs1[i].DebugString()<<endl;
  }
  std::cout << "Output tensor2 size:" << outputs2.size() << std::endl;
  for (std::size_t i = 0; i < outputs2.size(); i++) {
      std::cout << outputs2[i].DebugString()<<endl;
  }
 
  cout<<1<<endl;
  Tensor t1 = outputs1[0];                   // Fetch the first tensor
  cout<<2<<endl;
  cout<<t1.DebugString()<<endl;
  int ndim21 = t1.shape().dims();             // Get the dimension of the tensor
  cout<<3<<endl;
  cout<<ndim21<<endl;
  //auto tmap = t.tensor<float, 2>();        // Tensor Shape: [batch_size, target_class_num]
  auto tmap1 = t1.tensor<float, 2>();
  cout<<4<<endl;
  //cout<<tmap<<endl;
  int output_dim1 = t1.shape().dim_size(1);  // Get the target_class_num from 1st dimension
  cout<<5<<endl;
  cout<<output_dim1<<endl;
  std::vector<double> tout1;
  // Argmax: Get Final Prediction Label and Probability
  int output_class_id = -1;

  for (int j = 0; j < output_dim1; j++)
  {
        std::cout <<"distance"<<":" << tmap1(0, j) << "," ;
  //      if (tmap(0, j) >= output_prob) {
 //             output_class_id = j;
  //            output_prob = tmap(0, j);
   //        }
  }
  endTime=clock();
  double duatime;
  duatime=(double)((endTime-startTime)/CLOCKS_PER_SEC);
  cout<<"duration:"<<duatime<<"s"<<endl;
  return 0;


}
