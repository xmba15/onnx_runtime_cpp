#include "Utility.hpp"
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#ifdef ENABLE_TORCH
#include "torch/torch.h"
#include "torch/script.h"
#endif
#include "LISRD.hpp"
#include "SuperPoint.hpp"
#include <ort_utility/ort_utility.hpp>

std::vector<cv::KeyPoint> getKeyPointFromSift(const cv::Mat& img, int nfeatures, float contrastThreshold)
{
    std::vector<cv::KeyPoint> keypoints;
    // cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create(nfeatures);
    cv::Ptr<cv::Feature2D> siftDetector = cv::SIFT::create(nfeatures);
    siftDetector->detect(img, keypoints);
    std::vector<cv::KeyPoint> result;
    for (const auto& k : keypoints)
    {
      result.push_back(cv::KeyPoint(k.pt.y, k.pt.x, k.response));
    }
    return result;
};

std::vector<cv::KeyPoint> getKeyPoints(const std::vector<Ort::OrtSessionHandler::DataOutputType>& inferenceOutput,
                                       int borderRemove = 4, float confidenceThresh = 0.015)
{
  std::vector<int> detectorShape(inferenceOutput[0].second.begin() + 1, inferenceOutput[0].second.end()); // 定义变量

  cv::Mat detectorMat(detectorShape.size(), detectorShape.data(), CV_32F, inferenceOutput[0].first); // 65 x H/8 x W/8

  cv::Mat buffer;
  transposeNDWrapper(detectorMat, {1, 2, 0}, buffer);
  buffer.copyTo(detectorMat); // H/8 x W/8 x 65

  for (int i = 0; i < detectorShape[1]; ++i)
  {
    for (int j = 0; j < detectorShape[2]; ++j)
    {
      Ort::softmax(detectorMat.ptr<float>(i, j), detectorShape[0]); // .ptr<float>(i, j)
    }
  }
  // same as python code dense[:-1, :, :]
  detectorMat =
    detectorMat({cv::Range::all(), cv::Range::all(), cv::Range(0, detectorShape[0] - 1)}).clone(); // H/8 x W/8 x 64
  detectorMat = detectorMat.reshape(1, {detectorShape[1], detectorShape[2], 8, 8}); // H/8 x W/8 x 8 x 8 create one dims
  transposeNDWrapper(detectorMat, {0, 2, 1, 3}, buffer);
  buffer.copyTo(detectorMat);                                                         // H/8 x 8 x W/8 x 8
  detectorMat = detectorMat.reshape(1, {detectorShape[1] * 8, detectorShape[2] * 8}); // H x W

  std::vector<cv::KeyPoint> keyPoints;
  for (int i = borderRemove; i < detectorMat.rows - borderRemove; ++i)
  {
    auto rowPtr = detectorMat.ptr<float>(i); // 指针
    for (int j = borderRemove; j < detectorMat.cols - borderRemove; ++j)
    {
      if (rowPtr[j] > confidenceThresh) 
      {
        cv::KeyPoint keyPoint;
        keyPoint.pt.x = j;
        keyPoint.pt.y = i;
        keyPoint.response = rowPtr[j];    //
        keyPoints.emplace_back(keyPoint); 
      }
    }
  }
  return keyPoints;
}

std::vector<cv::KeyPoint> nmsFast_good(std::vector<cv::KeyPoint>& in_corners, int H, int W, int dist_thresh = 4) 
{
  // Create a grid sized HxW. Assign each corner location a 1, rest
  // are zeros.
  cv::Mat grid = cv::Mat::zeros(H, W, CV_32S); // CV_32S == int
  std::vector<int> inds(H * W);
  // Sort by confidence and round to nearest int.
  std::sort(in_corners.begin(), in_corners.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
      return a.response > b.response;});
  // Rounded corners.
  std::vector<cv::Point> rcorners;
  for (const auto& corner : in_corners) 
  {
      rcorners.emplace_back(cvRound(corner.pt.x), cvRound(corner.pt.y));
  }
  // Check for edge case of 0 or 1 corners.
  if (rcorners.empty()) 
  {
      return {};
  }
  if (rcorners.size() == 1) 
  {
      return {in_corners[0]};
  }
  // Initialize the grid.
  for (int i = 0; i < rcorners.size(); i++) 
  {
      grid.at<int>(rcorners[i]) = 1;
      inds[rcorners[i].y * W + rcorners[i].x] = i;
  }
  // Pad the border of the grid, so that we can NMS points near the border.
  int pad = dist_thresh;
  cv::copyMakeBorder(grid, grid, pad, pad, pad, pad, cv::BORDER_CONSTANT, 0);
  // Iterate through points, highest to lowest conf, suppress neighborhood.
  int count = 0;
  for (int i = 0; i < rcorners.size(); i++) 
  {
    // Account for top and left padding.
    cv::Point pt(rcorners[i].x + pad, rcorners[i].y + pad);
    if (grid.at<int>(pt) == 1) 
    { 
      // If not yet suppressed
      cv::KeyPoint new_kp = in_corners[inds[rcorners[i].y * W + rcorners[i].x]];
      new_kp.pt.x = rcorners[i].x - pad;
      new_kp.pt.y = rcorners[i].y - pad;
      in_corners[count++] = new_kp;
      // Suppress neighbors.
      for (int dx = -dist_thresh; dx <= dist_thresh; dx++) 
      {
        for (int dy = -dist_thresh; dy <= dist_thresh; dy++) 
        {
            grid.at<int>(pt.y + dy, pt.x + dx) = 0;
        }
      }
    }
  }
  in_corners.resize(count);
  // return in_corners;
  std::vector<cv::KeyPoint> result;
  for (const auto& k : in_corners)
  {
    result.push_back(cv::KeyPoint(k.pt.y, k.pt.x, k.response));
  }
  return result;
}

torch::Tensor keyPointsToGrid(const std::vector<cv::KeyPoint>& in_keypoints, const cv::Size& img_size) 
{
  //-----------------------------------------------------//
  // [k.pt[1], k.pt[0], k.response in python
  //-----------------------------------------------------//
  std::vector<float> keypoints_data;
  for (const auto& keypoint : in_keypoints) {
      keypoints_data.push_back(keypoint.pt.x);
      keypoints_data.push_back(keypoint.pt.y);
  }
  int n_points = in_keypoints.size();
  torch::Tensor keypoints_tensor = torch::from_blob(keypoints_data.data(), { n_points, 2 }, torch::kFloat32);
  torch::Tensor img_size_tensor = torch::tensor({ img_size.height, img_size.width }, torch::kFloat32);
  torch::Tensor points_tensor = keypoints_tensor * 2.0 / img_size_tensor - 1.0;
  torch::Tensor index = torch::tensor({ 1, 0 }, torch::dtype(torch::kLong));
  points_tensor = points_tensor.index_select(1, index);
  torch::Tensor grid_keypoints_tensor = points_tensor.view({ -1, n_points, 1, 2 }); // .item<float>(i, j, k, l)
  return grid_keypoints_tensor;
};

std::pair<torch::Tensor, torch::Tensor>
extractDescriptors(const std::vector<Ort::OrtSessionHandler::DataOutputType> &lisrd_outputs, 
                   const std::vector<cv::KeyPoint>& in_keypoints, const cv::Size& img_size)
{
  torch::Tensor grid_points = keyPointsToGrid(in_keypoints, img_size);
  torch::Tensor descs, meta_descs;
  std::vector<torch::Tensor> descs_vector, meta_descs_vector;
  for (size_t i = 0; i < lisrd_outputs.size(); i++)
  {
    Ort::OrtSessionHandler::DataOutputType output = lisrd_outputs.at(i);
    if (i < 4)
    {
      std::vector<int> descShape(output.second.begin(), output.second.end()); // 1 128 90 160 -- 
      cv::Mat desc(descShape.size(), descShape.data(), CV_32F, output.first);
      torch::Tensor tensor_desc = torch::from_blob(desc.data, {desc.size[0], desc.size[1], desc.size[2], desc.size[3]}, torch::kFloat32);
      torch::nn::functional::GridSampleFuncOptions sample_options;
      sample_options.align_corners(true);
      torch::Tensor smaple_desc = torch::nn::functional::grid_sample(tensor_desc, grid_points, sample_options);
      torch::nn::functional::NormalizeFuncOptions normal_options;
      normal_options.p(2);
      normal_options.dim(1);
      torch::Tensor normal_desc = torch::nn::functional::normalize(smaple_desc, normal_options);
      torch::Tensor squeeze_desc = torch::squeeze(normal_desc); //
      torch::Tensor trans_desc = torch::transpose(squeeze_desc, 0, 1);
      descs_vector.push_back(trans_desc);
    }
    else
    {
      std::vector<int> metadescShape(output.second.begin(),
                                     output.second.end()); // metadescShape.data() 表示数组的维度
      cv::Mat meta_desc(metadescShape.size(), metadescShape.data(), CV_32F, output.first);
      torch::Tensor tensor_meta_desc = torch::from_blob(meta_desc.data, {meta_desc.size[0], meta_desc.size[1], meta_desc.size[2], meta_desc.size[3]}, torch::kFloat32);
      torch::nn::functional::GridSampleFuncOptions sample_options;
      sample_options.align_corners(true);
      torch::Tensor smaple_meta_desc = torch::nn::functional::grid_sample(tensor_meta_desc, grid_points, sample_options);
      torch::nn::functional::NormalizeFuncOptions normal_options;
      normal_options.p(2);
      normal_options.dim(1);
      torch::Tensor normal_meta_desc = torch::nn::functional::normalize(smaple_meta_desc, normal_options);
      torch::Tensor squeeze_meta_desc = torch::squeeze(normal_meta_desc); //
      torch::Tensor trans_meta_desc = torch::transpose(squeeze_meta_desc, 0, 1);
      meta_descs_vector.push_back(trans_meta_desc);
    }
  }
  descs      = torch::stack(descs_vector, 1);
  meta_descs = torch::stack(meta_descs_vector, 1);
  return std::make_pair(descs, meta_descs);
}

torch::Tensor lisrdMatcher(torch::Tensor desc1, torch::Tensor desc2, torch::Tensor meta_desc1, torch::Tensor meta_desc2)
{
  torch::Tensor desc_weights = torch::einsum("nid,mid->nim", {meta_desc1, meta_desc2}); // 元素相乘 [keypoint1, 4, keypoint2]
  meta_desc1.reset();
  meta_desc2.reset();
  desc_weights = torch::softmax(desc_weights, 1);
  torch::Tensor desc_sims = torch::einsum("nid,mid->nim", {desc1, desc2}) * desc_weights;
  desc1.reset();
  desc2.reset();
  desc_weights.reset();
  desc_sims = torch::sum(desc_sims, 1);
  torch::Tensor nn12 = torch::argmax(desc_sims, 1);
  torch::Tensor nn21 = torch::argmax(desc_sims, 0);
  torch::Tensor ids1 = torch::arange(desc_sims.size(0), torch::dtype(torch::kLong));
  // desc_sims.reset();
  torch::Tensor mask = (ids1 == nn21.index_select(0, nn12));
  torch::Tensor mask_bool = mask.nonzero().squeeze(1);
  torch::Tensor t1 = torch::index_select(ids1, 0, mask_bool); //
  torch::Tensor t2 = torch::index_select(nn12, 0, mask_bool); //
  torch::Tensor matches = torch::stack({t1, t2}, 1);
  return matches;
}

std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>
filterOutliersRansac(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2)
{
  std::vector<cv::Point2f> kp1_pts, kp2_pts;
  for (const auto& kp : kp1)
    kp1_pts.emplace_back(kp.pt);
  for (const auto& kp : kp2)
    kp2_pts.emplace_back(kp.pt);
  std::vector<unsigned char> inliers(kp1_pts.size());
  cv::findHomography(kp1_pts, kp2_pts, cv::RANSAC, 3, inliers);
  std::vector<cv::KeyPoint> filtered_kp1, filtered_kp2;
  for (int i = 0; i < inliers.size(); i++)
  {
    if (inliers[i])
    {
      filtered_kp1.push_back(kp1[i]);
      filtered_kp2.push_back(kp2[i]);
    }
  }
  return std::make_pair(filtered_kp1, filtered_kp2);
}

void plot_keypoints(const cv::Mat &img, const std::vector<cv::KeyPoint>& kpts, const std::vector<cv::Scalar>& colors, float ps) 
{
for (int i = 0; i < kpts.size(); i++) 
  {
      cv::KeyPoint k = kpts[i];
      cv::Scalar c = colors[i];
      // cv::circle(img, k.pt, ps, c, -1);
      cv::Point2f pt = k.pt;
      cv::circle(img, cv::Point2f(pt.y, pt.x), ps, c, -1);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::cerr
        << "Usage: [apps] [path/to/onnx/super/point] [path/to/onnx/lisrd] [path/to/image1] [path/to/image2]"
        << std::endl;
    return EXIT_FAILURE;
  }

  const std::string superpoint_model_path = argv[1];
  const std::string lisrd_model_path = argv[2];

  cv::Mat bgr1  = cv::imread(argv[3], cv::IMREAD_COLOR);
  cv::Mat bgr2  = cv::imread(argv[4], cv::IMREAD_COLOR);
  cv::Mat gray1 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
  cv::Mat gray2 = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
  cv::Mat rgb1, rgb2, resized_img1, resized_img2;
  cv::cvtColor(bgr1, rgb1, cv::COLOR_BGR2RGB);
  cv::cvtColor(bgr2, rgb2, cv::COLOR_RGB2BGR);
  cv::resize(rgb1, rgb1, cv::Size(480, 640), cv::INTER_CUBIC);
  cv::resize(rgb2, rgb2, cv::Size(480, 640), cv::INTER_CUBIC);
  //-----------------------------------------------------//
  // keypoint from sift
  //-----------------------------------------------------//
  // std::vector<cv::KeyPoint> keypoint1 = getKeyPointFromSift(rgb1, 1500, 0.04);
  // std::vector<cv::KeyPoint> keypoint2 = getKeyPointFromSift(rgb2, 1500, 0.04);

  //-----------------------------------------------------//
  // keypoint from superpoint and inference
  //-----------------------------------------------------//
  Ort::SuperPoint superpoint1(superpoint_model_path, 0, std::vector<std::vector<int64_t>>{
                              {1, gray1.channels(), gray1.size().height, gray1.size().width}});
  std::vector<float> superpoint_input1(gray1.channels() * gray1.size().width * gray1.size().height);
  superpoint1.preprocess(superpoint_input1.data(), gray1.data, gray1.size().height, gray1.size().width, gray1.channels());
  std::vector<Ort::OrtSessionHandler::DataOutputType> superpoint_output1 = superpoint1({superpoint_input1.data()});
  std::vector<cv::KeyPoint> keypoint1 = getKeyPoints(superpoint_output1);
  keypoint1 = nmsFast_good(keypoint1, gray1.size().height, gray1.size().width);
  

  Ort::SuperPoint superpoint2(superpoint_model_path, 0, std::vector<std::vector<int64_t>>{
                              {1, gray2.channels(), gray2.size().height, gray2.size().width}});
  std::vector<float> superpoint_input2(gray2.channels() * gray2.size().width * gray2.size().height);
  superpoint2.preprocess(superpoint_input2.data(), gray2.data, gray2.size().height, gray2.size().width, gray2.channels());
  std::vector<Ort::OrtSessionHandler::DataOutputType> superpoint_output2 = superpoint2({superpoint_input2.data()});
  std::vector<cv::KeyPoint> keypoint2 = getKeyPoints(superpoint_output2);
  keypoint2 = nmsFast_good(keypoint2, gray2.size().height, gray2.size().width);
  //-----------------------------------------------------//
  // descriptor from lisrd and inference
  //-----------------------------------------------------//
  auto start = std::chrono::steady_clock::now();
  Ort::Lisrd lisrd1(lisrd_model_path, 0, std::vector<std::vector<int64_t>>{
                   {1, rgb1.channels(), rgb1.size().height, rgb1.size().width}});
  std::vector<float> lisrd_input1(rgb1.channels() * rgb1.size().width * rgb1.size().height); // define float vector and shape
  lisrd1.Preprocess(lisrd_input1.data(), rgb1.data, rgb1.size().height, rgb1.size().width, rgb1.channels());
  std::vector<Ort::OrtSessionHandler::DataOutputType> lisrd_output1 = lisrd1({lisrd_input1.data()});

  Ort::Lisrd lisrd2(lisrd_model_path, 0,std::vector<std::vector<int64_t>>{
                   {1, rgb1.channels(), rgb1.size().height, rgb1.size().width}});
  std::vector<float> lisrd_input2(rgb2.channels() * rgb2.size().width * rgb2.size().height); // define float vector and shape
  lisrd2.Preprocess(lisrd_input2.data(), rgb2.data, rgb2.size().height, rgb2.size().width, rgb2.channels());
  std::vector<Ort::OrtSessionHandler::DataOutputType> lisrd_output2 = lisrd2({lisrd_input2.data()});

  std::pair<torch::Tensor, torch::Tensor> result1 = extractDescriptors(lisrd_output1, keypoint1, rgb1.size());
  std::pair<torch::Tensor, torch::Tensor> result2 = extractDescriptors(lisrd_output2, keypoint2, rgb2.size());
  torch::Tensor matches = lisrdMatcher(result1.first, result2.first, result1.second, result2.second);
  //--------------------------------------------------------------//
  // torch::Tensor --> cv::Mat
  //--------------------------------------------------------------//
  cv::Mat matches_mat(matches.size(0), matches.size(1), CV_32SC1);
  matches = matches.to(at::kInt);
  auto matches_accessor = matches.accessor<int32_t, 2>();
  for (int i = 0; i < matches.size(0); i++)
  {
      for (int j = 0; j < matches.size(1); j++)
      {
          matches_mat.at<int>(i, j) = matches_accessor[i][j];
      }
  }
  std::cout << "cv matches size : " << matches_mat.size() << std::endl;
  //-------------------------------------------------------------//
  // kp1[matches[:, 0]][:, [1, 0]], kp2[matches[:, 1]][:, [1, 0]]
  //-------------------------------------------------------------//
  std::vector<cv::KeyPoint> matched_kp1, matched_kp2;
  for (int i = 0; i < matches_mat.rows; i++)
  {
      int idx1 = matches_mat.at<int>(i, 0);
      int idx2 = matches_mat.at<int>(i, 1);
      cv::KeyPoint kp1_temp = keypoint1[idx1];
      cv::KeyPoint kp2_temp = keypoint2[idx2];
      // Swap x and y coordinates
      std::swap(kp1_temp.pt.x, kp1_temp.pt.y);
      std::swap(kp2_temp.pt.x, kp2_temp.pt.y);
      matched_kp1.push_back(kp1_temp);
      matched_kp2.push_back(kp2_temp);
  }
  
  //-------------------------------------------------------------//
  // python in filter_outliers_ransac
  //-------------------------------------------------------------//
  std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> filterKeyPoints = filterOutliersRansac(matched_kp1, matched_kp2);
  std::vector<cv::KeyPoint> filtered_kp1, filtered_kp2;

  std::vector<cv::DMatch> matches_info;
  for (int i = 0; i < filterKeyPoints.first.size(); i++)
  {
      matches_info.push_back(cv::DMatch(i, i, 0));
  }

  cv::Mat matchesImage;
  std::cout << "filterKeyPoints first: " << filterKeyPoints.first.size() << " filterKeyPoints second: " << filterKeyPoints.second.size() << std::endl;
  cv::drawMatches(bgr1, filterKeyPoints.first, bgr2, filterKeyPoints.second, matches_info, matchesImage,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imwrite("Lisrd_good_matches.jpg", matchesImage);
  // cv::imshow("Lisrd_good_matches", matchesImage);
  // cv::waitKey();

  return EXIT_SUCCESS;
}
