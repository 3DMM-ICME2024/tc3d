// zhangcheng 2015.9.12
// convert image pair and label into siamese network type
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "leveldb/db.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "opencv2/opencv.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace std;
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");


static bool ReadImageToMemory(const string& FileName, const int Height,
                              const int Width, char *Pixels)
{
    // read image
    //cv::Mat OriginImage = cv::imread(FileName, cv::IMREAD_GRAYSCALE);
    cv::Mat ResizeImage = cv::imread(FileName, cv::IMREAD_GRAYSCALE);
    //CHECK(OriginImage.data) << "Failed to read the image.\n";


    // resize the image
    //cv::Mat ResizeImage;
    //cv::resize(OriginImage, ResizeImage, cv::Size(Width, Height));
    //CHECK(ResizeImage.rows == Height) << "The heighs of Image is no equal to the input height.\n";
    //CHECK(ResizeImage.cols == Width) << "The width of Image is no equal to the input width.\n";
    //CHECK(ResizeImage.channels() == 1) << "The channel of Image is no equal to one.\n";

    //LOG(INFO) << "The height of image is " << ResizeImage.rows << "\n";
    //LOG(INFO) << "The width of image is " << ResizeImage.cols << "\n";
    //LOG(INFO) << "The channels of image is " << ResizeImage.channels() << "\n";

    // copy the image data to Pixels
    for (int HeightIndex = 0; HeightIndex < Height; ++HeightIndex)
    {
        const uchar* ptr = ResizeImage.ptr<uchar>(HeightIndex);
        int img_index = 0;
        for (int WidthIndex = 0; WidthIndex < Width; ++WidthIndex)
        {
            for (int ChannelIndex = 0; ChannelIndex < ResizeImage.channels(); ++ChannelIndex)
            {
                int datum_index = (ChannelIndex * Height + HeightIndex) * Width + WidthIndex;
                *(Pixels + datum_index) = static_cast<char>(ptr[img_index++]);
            }
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    //::google::InitGoogleLogging(argv[0]);


    ifstream in1(argv[2]);
    ifstream in0(argv[3]);

    // set height and width
    int resize_height = 128;//std::max<int>(0, FLAGS_resize_height);
    int resize_width = 88;//std::max<int>(0, FLAGS_resize_width);

    // / Open leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
        options, argv[4], &db);
    CHECK(status.ok()) << "Failed to open leveldb " << argv[3]
        << ". Is it already existing?";


    // save to leveldb
    // Storing to leveldb
    std::string root_folder(argv[1]);
    char* Pixels = new char[2 * resize_height * resize_width];
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    std::string value;

    caffe::Datum datum;
    datum.set_channels(2);  // one channel for each image in the pair
    datum.set_height(resize_height);
    datum.set_width(resize_width);

    string img1,img2;
    int label1,label2,tag;
    int LineIndex = 0;    
    vector<pair<string, string> > line_0;
    vector<pair<string, string> > line_1;
    cout << "Read the data...\n";
    while(in0 >> img1 >> img2 >> label1 >> label2 >> tag){
	line_0.push_back(make_pair(img1,img2));
    }
    while(in1 >> img1 >> img2 >> label1 >> label2 >> tag){
	line_1.push_back(make_pair(img1,img2));
    }
    cout << "Shuffle the data...\n";
    shuffle(line_0.begin(), line_0.end());
    shuffle(line_1.begin(), line_1.end());

    //for (int LineIndex = 0; LineIndex < lines.size(); LineIndex++)
    for (int i = 0; i < line_1.size(); i++)
    {
        //int PairIndex = caffe::caffe_rng_rand() % lines.size();
	cout << line_1[i].first << " " << line_1[i].second << endl;
        char* FirstImagePixel = Pixels;
	    ReadImageToMemory(root_folder + line_1[i].first, resize_height, resize_width, FirstImagePixel);
	    char *SecondImagePixel = Pixels + resize_width * resize_height;
        ReadImageToMemory(root_folder + line_1[i].second, resize_height, resize_width, SecondImagePixel);
        datum.set_data(Pixels, 2 * resize_height * resize_width);
        datum.set_label(1);
        // serialize datum to string
        datum.SerializeToString(&value);
	    snprintf(key, kMaxKeyLength, "%08d", LineIndex);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
	    LineIndex++;

	    cout << line_0[i].first << " " << line_0[i].second << endl;
	    char* F = Pixels;
	    ReadImageToMemory(root_folder + line_0[i].first, resize_height, resize_width, F);
        char* S = Pixels + resize_width * resize_height;
        ReadImageToMemory(root_folder + line_0[i].second, resize_height, resize_width, S);
        datum.set_data(Pixels, 2 * resize_height * resize_width);
        datum.set_label(0);
        datum.SerializeToString(&value);
        snprintf(key, kMaxKeyLength, "%08d", LineIndex);
        db->Put(leveldb::WriteOptions(), std::string(key), value);
        LineIndex++;
    }
    cout << "LindIndex:" << LineIndex << endl;
    delete db;
    delete[] Pixels;

    return 0;
}
