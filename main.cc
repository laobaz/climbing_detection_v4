

#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <utility>
#include <stdarg.h>

#include <sys/stat.h>

#include "code/object_detector.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

static std::string DirName(const std::string &filepath)
{
  auto pos = filepath.rfind(OS_PATH_SEP);
  if (pos == std::string::npos)
  {
    return "";
  }
  return filepath.substr(0, pos);
}

static bool PathExists(const std::string &path)
{
#ifdef _WIN32
  struct _stat buffer;
  return (_stat(path.c_str(), &buffer) == 0);
#else
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
#endif // !_WIN32
}

static void MkDir(const std::string &path)
{
  if (PathExists(path))
    return;
  int ret = 0;
#ifdef _WIN32
  ret = _mkdir(path.c_str());
#else
  ret = mkdir(path.c_str(), 0755);
#endif // !_WIN32
  if (ret != 0)
  {
    std::string path_error(path);
    path_error += " mkdir failed!";
    throw std::runtime_error(path_error);
  }
}

static void MkDirs(const std::string &path)
{
  if (path.empty())
    return;
  if (PathExists(path))
    return;

  MkDirs(DirName(path));
  MkDir(path);
}

static inline float intersection_area(const PaddleDetection::ObjectResult &a, const PaddleDetection::ObjectResult &b)
{
  int left = std::max(a.rect[0], b.rect[0]);
  int right = std::min(a.rect[2], b.rect[2]);
  int top = std::max(a.rect[1], b.rect[1]);
  int bottom = std::min(a.rect[3], a.rect[3]);

  // 计算宽度和高度
  int width = right - left;
  int height = bottom - top;

  if (width > 0 && height > 0)
  {
    return (float)(width * height);
  }
  else
  {
    return 0; // 没有交集
  }
}
static inline float chepaiobj_areas(const PaddleDetection::ObjectResult &a)
{
  return (a.rect[3] - a.rect[1]) * (a.rect[2] - a.rect[0]);
}

bool predict_climb_or(std::vector<std::pair<PaddleDetection::ObjectResult,
                                            std::pair<PaddleDetection::ObjectResult, PaddleDetection::ObjectResult>>>
                          F_p_F,
                      std::vector<PaddleDetection::ObjectResult> climb_result)
{
  for (int i = 0; i < F_p_F.size(); i++)
  {
    // 判断人是否在攀登物上里面
    for (int j = 0; j < climb_result.size(); j++)
    {
      float inter_area = intersection_area(F_p_F[i].first, climb_result[j]);
      // std::cout<<"person's inter_area is "<<inter_area<<std::endl;
      float chepaiobj_area = chepaiobj_areas(F_p_F[i].first);
      // std::cout<<"person's chepaiobj_area is "<<chepaiobj_area<<std::endl;
      double IOU = inter_area / chepaiobj_area;
      // std::cout<<"person's IOU is "<<IOU<<std::endl;

      if (inter_area / chepaiobj_area > 0.7)
      {
        //[0][1]为左上角x,y[2][3]为右下角x,y
        // int w = results[i].rect[2] - results[i].rect[0];
        //  int h = results[i].rect[3] - results[i].rect[1];
        //  int person_foot_x= F_p_F[i].second.second.rect[0];
        //  int person_foot_y=F_p_F[i].second.second.rect[1];
        //  int person_foot_width=F_p_F[i].second.second.rect[2]-F_p_F[i].second.second.rect[0];
        //  int person_foot_height=F_p_F[i].second.second.rect[3]-F_p_F[i].second.second.rect[1];

        // 高的攀登物情况 脚
        if (climb_result[j].rect[3] > F_p_F[i].second.second.rect[3]    // 脚的y轴在墙上，
            && climb_result[j].rect[0] < F_p_F[i].second.second.rect[0] // 脚在墙的范围内
            && climb_result[j].rect[2] > F_p_F[i].second.second.rect[2])
        {
          return true;
        }
        // 头
        else if (climb_result[j].rect[0] - 15 < F_p_F[i].second.first.rect[0] // 头超过墙上边，
                 && climb_result[j].rect[0] < F_p_F[i].second.first.rect[0]   // 头在墙的范围内
                 && climb_result[j].rect[2] > F_p_F[i].second.first.rect[2])
        {
          return true;
        }
        else
        {
          continue;
        }
      }
      else
      {
        continue;
      }
    }
  }
  return false;
}

void predict_video(cv::Mat flame,
                   const int batch_size,
                   const double threshold,
                   PaddleDetection::ObjectDetector *det,
                   int frameCount,
                   const std::string &output_dir,
                   cv::VideoWriter &outputVideo,
                   const int fps,
                   const int frame_width,
                   const int frame_height)
{
  std::vector<PaddleDetection::ObjectResult> result;
  std::vector<int> bbox_num; // 每张图片的object数目
  std::vector<double> det_times;
  bool is_rbox = false;

  // std::cout << "Predict start" << std::endl;
  std::vector<cv::Mat> imgs;
  imgs.push_back(flame);
  det->Predict(imgs, 0, 1, &result, &bbox_num, &det_times);

  // std::cout << "Predict end" << std::endl;
  //  get labels and colormap
  auto labels = det->GetLabelList();
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());

  int item_start_idx = 0;

  cv::Mat &im = flame;
  std::vector<PaddleDetection::ObjectResult> im_result;
  std::vector<PaddleDetection::ObjectResult> face_result;
  std::vector<PaddleDetection::ObjectResult> foot_result;
  std::vector<PaddleDetection::ObjectResult> person_result;
  std::vector<PaddleDetection::ObjectResult> climb_result;
  int detect_num = 0;
  // 过滤object部分
  for (int j = 0; j < bbox_num[0]; j++)
  {
    PaddleDetection::ObjectResult item = result[item_start_idx + j];

    if (!isfinite(item.confidence))
    {
      // ANNIWOLOG(INFO) <<logstr<<":get unexpected infinite value!!" ;
      // std::cout << ":get unexpected infinite value!!" << std::endl;
    }

    if (item.confidence < threshold || item.class_id == -1)
    {
      continue;
    }
    detect_num += 1;
    im_result.push_back(item);
    if (item.class_id == 7)
    {
      foot_result.push_back(item);
    }
    else if (item.class_id == 6)
    {
      face_result.push_back(item);
    }

    else if (item.class_id == 0)
    {
      person_result.push_back(item);
    }
    else if (item.class_id != 3)
    {
      climb_result.push_back(item);
    }
  }
  cv::Mat vis_img = PaddleDetection::VisualizeResult(
      im, im_result, labels, colormap, is_rbox);

  std::vector<std::pair<PaddleDetection::ObjectResult,
                        std::pair<PaddleDetection::ObjectResult, PaddleDetection::ObjectResult>>>
      F_p_F;
  bool have_person = false;
  for (auto pobj : person_result)
  {
    PaddleDetection::ObjectResult face_and_p_obj;
    bool face_and_p = false;
    for (auto face_obj : face_result)
    {

      float inter_area = intersection_area(pobj, face_obj);
      float chepaiobj_area = chepaiobj_areas(face_obj);
      double IOU = inter_area / chepaiobj_area;
      if (inter_area / chepaiobj_area > 0.8)
      {
        face_and_p_obj = face_obj;
        face_and_p = true;
        break;
      }
    }
    PaddleDetection::ObjectResult foot_and_p_obj;
    bool foot_and_p = false;
    for (auto foot_obj : foot_result)
    {
      float inter_area = intersection_area(pobj, foot_obj);
      float chepaiobj_area = chepaiobj_areas(foot_obj);
      if (inter_area / chepaiobj_area > 0.8)
      {
        foot_and_p_obj = foot_obj;
        foot_and_p = true;
        break;
      }
    }
    if (foot_and_p && face_and_p)
    {
      // std::vector<std::pair<PaddleDetection::ObjectResult,
      //       std::pair<PaddleDetection::ObjectResult>>> F_p_F;
      F_p_F.push_back(std::make_pair(pobj, std::make_pair(face_and_p_obj, foot_and_p_obj)));
      have_person = true;
      // for(auto obj:F_p_F){
      //   //std::cout<<obj.first.confidence<<"  "<<obj.second.first.confidence<<"  "<<obj.second.second.confidence<<std::endl;
      // }
      // std::cout<<"have a complete person"<<std::endl;
    }
    else
    {
      continue;
    }
  }
  if (have_person)
  {
    // 判断是否有登高的危险
    // std::cout<<"staring predict warn?"<<std::endl;
    bool warn = predict_climb_or(F_p_F, climb_result);

    // 输出部分
    if (warn)
    {

      cv::Scalar red_color(0, 0, 255); // 红色
      int thickness = 2;               // 框的厚度
      cv::rectangle(vis_img, cv::Point(10, 10), cv::Point(frame_width - 10, frame_height - 10), red_color, thickness);

      // 标签文本和字体设置
      std::string label = "Warning";
      int font_face = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 1.0;
      int font_thickness = 2;
      cv::Scalar font_color(0, 0, 255); // 白色字体
      int baseline = 0;

      // 计算标签文字大小，以便确定文字放置位置
      cv::Size text_size = cv::getTextSize(label, font_face, font_scale, font_thickness, &baseline);
      // 文字位置：放置在矩形的左上角，确保文字不超出图像边界
      cv::Point text_org((frame_width - text_size.width) / 2, text_size.height + 10);

      // 在图像上绘制标签
      cv::putText(vis_img,
                  label,
                  text_org,
                  font_face,
                  font_scale,
                  font_color,
                  font_thickness);
      if (frameCount % fps == 0)
      {
        std::cout << "warn!!! maybe having person  is climing" << std::endl;
        std::string img_path = std::to_string(frameCount / fps);
        // Visualization resu

        std::string output_path(output_dir);
        if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1)
        {
          output_path += OS_PATH_SEP;
        }
        output_path +=
            (std::string("warn_") + img_path + ".jpg");

        cv::imwrite(output_path, vis_img);

        printf("Visualized output saved as %s\n", output_path.c_str());
      }
    }
  }
  else
  {
    // std::cout<<"no have a complete person"<<std::endl;
  }
  outputVideo.write(vis_img);
}

int main(int argc, char **argv)
{

  const char *engine_file_path_in = "trt_model_xz/rtdetr_r50vd_6x_coco_xz_new.trt";
  // const char* engine_file_path_in="./trt_model_helmet/rtdetr_r50vd_6x_coco.trt";
  const char *config_file_path_in = "trt_model_xz/infer_cfg.yml";

  PaddleDetection::ObjectDetector det(config_file_path_in, engine_file_path_in);

  // std::vector<std::string> all_img_paths;
  // all_img_paths.push_back(input_image_path);

  //   ///////////////
  // std::vector<cv::String> files;
  // cv::glob("images/", files);
  // for (int i = 0; i < files.size(); i++)
  // {
  //   std::string file = files[i];
  //   printf("===file: %s\n", file.c_str());
  //   all_img_paths.push_back(file);
  // }
  // //////////////
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <VideoFilePath>" << "<out_imgs_Path>" << std::endl;
    return -1;
  }
  std::string videoPath = argv[1];
  std::string outimgsPath = argv[2];

  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened())
  {
    std::cerr << "Error opening video stream or file" << std::endl;
    return -1;
  }
  else
  {
    std::cout << "vdeo exists: " << videoPath << std::endl;
  }
  struct stat info;

  if (stat(argv[2], &info) != 0)
  {
    std::cerr << "Cannot access " << argv[2] << std::endl;
    return 1;
  }
  else if (info.st_mode & S_IFDIR)
  {
    std::cout << "Directory exists: " << argv[2] << std::endl;
  }
  else
  {
    std::cerr << argv[2] << " is not a directory." << std::endl;
    return 1;
  }

  // 获取视频帧率
  static double fps = static_cast<double>(cap.get(cv::CAP_PROP_FPS));
  static int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  static int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  int frameInterval = static_cast<int>(fps); // 每秒处理一帧

  cv::Mat frame;
  int frameCount = 0;

  cv::VideoWriter outputVideo("output_with_detections.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), frameInterval, cv::Size(frame_width, frame_height));

  while (cap.read(frame))
  {
    // 检查是否是该处理的帧
    // 在这里处理每秒的帧
    // std::cout << "Processing frame at " << frameCount / fps << " seconds." << std::endl;

    predict_video(frame, 1, 0.5, &det, (int)(frameCount), outimgsPath, outputVideo, frameInterval, frame_width, frame_height);

    // if (cv::waitKey(30) >= 0)
    // break; // 按下任意键退出

    frameCount++;
  }
  cap.release();
  outputVideo.release();
  return 0;
}
