#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\video\tracking.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include <iomanip>
#include <set>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

static const double pi = 3.14159265358979323846;
int maxCorners = 1000;
double qualityLevel = 0.01;
double minDistance = 10;
int blockSize = 3;
bool useHarrisDetector = false;
double k_harris = 0.04;

int maskSize = 10;

inline static double square(int a)

{

	return a * a;

}

void refresh_features(Mat gray, vector<Point2f> &flow_c);
void draw_opticalFlow(Point2f corners, Point2f flow_corners, Mat img, CvScalar line_color);
bool CheckCoherentRotation(cv::Mat_<double>& R);
Mat_<double> LinearLSTriangulation(
	Point3d u,//homogenous image point (u,v,1)
	Matx34d P,//camera 1 matrix
	Point3d u1,//homogenous image point in 2nd camera
	Matx34d P1//camera 2 matrix
	);

int main(int argc, char* argv[])
{
	
	double tick = getTickCount();
	Mat image = imread("test118.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat image2 = imread("test123.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat image3;
	image2.copyTo(image3);
	
	//no calibration matrix file - mockup calibration
	Size img_size = image.size();
	double max_w_h = MAX(img_size.height, img_size.width);
	Mat K = (cv::Mat_<double>(3, 3) << max_w_h, 0, img_size.width / 2.0,
		0, max_w_h, img_size.height / 2.0,
		0, 0, 1);
	Mat Kinv;
	invert(K, Kinv);

	namedWindow("hihi", WINDOW_NORMAL);
	//namedWindow("hoho", WINDOW_NORMAL);

	if (image.empty() || image2.empty()) //check whether the image is loaded or not
	{
		std::cout << "Error : Image cannot be loaded..!!" << std::endl;
		system("pause"); //wait for a key press
		return -1;
	}
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "before gray : " << tick << "s" << endl;
	tick = getTickCount();
	Mat image_gray, image_gray2;
	cvtColor(image, image_gray, CV_BGR2GRAY);
	cvtColor(image2, image_gray2, CV_BGR2GRAY);
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "after gray : " << tick << "s" << endl;
	tick = getTickCount();
	Mat mask(image_gray.size(), CV_8UC1, Scalar(255));
	vector< Point2f > corners, corners2, flow_corners;
	cout << "hihihihi" << endl;
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "BEFORE GFTT : " << tick << "s" << endl;
	tick = getTickCount();
	goodFeaturesToTrack(image_gray, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k_harris);
	goodFeaturesToTrack(image_gray2, corners2, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k_harris);
	//cout << "** Number of corners detected in image1: " << corners.size() << endl;
	//cout << "** Number of corners detected in image2: " << corners2.size() << endl;
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "processing time after goodFeature : " << tick << "s" << endl;
	tick = getTickCount();

	vector<uchar> status;
	vector<float> err;
	Size optical_flow_window = cvSize(3, 3);
	TermCriteria optical_flow_termination_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3);

	calcOpticalFlowPyrLK(image_gray, image_gray2, corners, flow_corners, status, err, optical_flow_window, 5, optical_flow_termination_criteria, 0, 0.001);
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "processing time after optical flow : " << tick << "s" << endl;
	tick = getTickCount();


	cout << "** Number of corners with optical flow in image2: " << corners.size() << endl;

	// First, filter out the points with high error
	vector<Point2f>right_points_to_find;
	vector<int>right_points_to_find_back_index;
	for (unsigned int i = 0; i<status.size(); i++) {
		if (status[i] && err[i] < 10.0) {
			// Keep the original index of the point in the
			// optical flow array, for future use
			right_points_to_find_back_index.push_back(i);
			// Keep the feature point itself
			right_points_to_find.push_back(flow_corners[i]);
			
			draw_opticalFlow(corners[i], flow_corners[i], image2, CV_RGB(255, 0, 0));

		}
		else {
			status[i] = 0; // a bad flow
		}
	}
	cout << "after optical flow status : " << right_points_to_find.size() << "/" << status.size() << endl;
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "processing time after optical flow status: " << tick << "s" << endl;
	tick = getTickCount();
	//imshow("hihi", image2);
	// for each right_point see which detected feature it belongs to
	Mat right_points_to_find_flat = Mat(right_points_to_find).reshape(1, right_points_to_find.size()); //flatten array
	Mat right_features_flat = Mat(corners2).reshape(1, corners2.size());
	// Look around each OF point in the right image
	// for any features that were detected in its area
	// and make a match.
	BFMatcher matcher(CV_L2);
	vector<vector<DMatch>>nearest_neighbors;
	matcher.radiusMatch(right_points_to_find_flat, right_features_flat, nearest_neighbors, 4.0f);
	// Check that the found neighbors are unique (throw away neighbors
	// that are too close together, as they may be confusing)
	vector<DMatch> matches;
	std::set<int>found_in_right_points; // for duplicate prevention
	for (int i = 0; i<nearest_neighbors.size(); i++) {
		DMatch _m;
		if (nearest_neighbors[i].size() == 1) {
			_m = nearest_neighbors[i][0]; // only one neighbor
		}
		else if (nearest_neighbors[i].size()>1) {
			// 2 neighbors – check how close they are
			double ratio = nearest_neighbors[i][0].distance /
				nearest_neighbors[i][1].distance;
			if (ratio < 0.7) { // not too close
							   // take the closest (first) one
				_m = nearest_neighbors[i][0];
			}
			else { // too close – we cannot tell which is better
				continue; // did not pass ratio test – throw away
			}
		}
		else {
			continue; // no neighbors... :(
		}
		// prevent duplicates
		if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.
			end()) {
			// The found neighbor was not yet used:
			// We should match it with the original indexing
			// ofthe left point
			_m.queryIdx = right_points_to_find_back_index[_m.queryIdx];
			matches.push_back(_m); // add this match
			found_in_right_points.insert(_m.trainIdx);
		}
	}
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "processing time after knn : " << tick << "s" << endl;
	cout << "pruned " << matches.size() << " / " << nearest_neighbors.size() << " matches" << endl;
	tick = getTickCount();
	//vector<Point2f> pts1;
	//for (int i = 0; i < matches.size(); i++) {
	//	pts1.push_back(right_points_to_find[matches[i].queryIdx]);
	//}
	vector<Point2f> imgpts1, imgpts2;
	for (unsigned int i = 0; i<matches.size(); i++) {
		imgpts1.push_back(corners[matches[i].queryIdx]);
		imgpts2.push_back(flow_corners[matches[i].queryIdx]);
		draw_opticalFlow(corners[matches[i].queryIdx], flow_corners[matches[i].queryIdx], image2, CV_RGB(0, 0, 255));
	}
	
	vector<uchar> stat(imgpts1.size());
	Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 0.1, 0.99, stat);
	cout << "F keeping " << countNonZero(stat) << " / " << stat.size() << endl;
	vector<Point2f> imgpts_good1, imgpts_good2;
	vector<int>imgpts_back_index;
	for (unsigned int i = 0; i<matches.size(); i++) {
		if (stat[i]) {
			imgpts_good1.push_back(imgpts1[i]);
			imgpts_good2.push_back(imgpts2[i]);
			imgpts_back_index.push_back(matches[i].queryIdx);
			draw_opticalFlow(corners[matches[i].queryIdx], flow_corners[matches[i].queryIdx], image2, CV_RGB(0, 255, 0));
		}
	}
	
	Mat_<double> E = K.t() * F * K;
	Matx34d P, P1;
	SVD svd(E, CV_SVD_MODIFY_A);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_w = svd.w;
	Matx33d W(0, -1, 0,
		1, 0, 0,
		0, 0, 1);
	Mat_ <double> R = svd_u * Mat(W) * svd_vt;
	Mat_ <double> t = svd_u.col(2);

	if (!CheckCoherentRotation(R)) {
		cout << "resulting rotation is no coherent" << endl;
		P1 = 0;
		return 0;
	}
	P = Matx34d(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);

	P1 = Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2));
	vector<double> reproj_error;
	vector<Point3d> pointcloud;
	Mat_<double> KP1 = K*Mat(P1); // 3x3 * 3x4 => 3*4
	cout << "P1 :" << P1 << endl;
	//cout << "Mat P1 :" << Mat(P1) << endl;
	for (unsigned int i = 0; i < imgpts_good1.size(); i++) {
		// convert to normalized homogenous coordinates
		Point3d u(imgpts_good1[i].x, imgpts_good1[i].y,1.0);
		Mat_<double> um = Kinv*Mat_<double>(u);
		u = um.at<Point3d>(0);
		//cout << "u :" << u << " , um :" << um << endl;
		Point3d u1(imgpts_good2[i].x, imgpts_good2[i].y, 1.0);
		Mat_<double> um1 = Kinv*Mat_<double>(u1);
		u1 = um1.at<Point3d>(0);
		// triangulate
		Mat_<double> X = LinearLSTriangulation(u, P, u1, P1);
		//cout << X << endl;
		//cout << KP1 << endl;
		// calculate reprojection error
		Mat_<double> xPt_img = KP1*X;
		//cout << "hihi" << endl;
		Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));
		reproj_error.push_back(norm(xPt_img_ - imgpts_good2[i]));
		//store 3D point
		pointcloud.push_back(Point3d(X(0), X(1), X(2)));
	}
	
	tick = ((double)getTickCount() - tick) / getTickFrequency();
	cout << "processing time after triang : " << tick << "s" << endl;
	Scalar mse = mean(reproj_error);
	cout << "Done. (" << pointcloud.size() << "points, mean reproj err = " << mse[0] << ")" << endl;
	for (unsigned int i = 0; i < pointcloud.size(); i++) {
		string x, y, z;
		stringstream streamx, streamy, streamz;
		streamx << fixed << setprecision(2) << pointcloud[i].x;
		x = streamx.str();
		streamy << fixed << setprecision(2) << pointcloud[i].y;
		y = streamy.str();
		streamz << fixed << setprecision(2) << pointcloud[i].z;
		z = streamz.str();
		String txt = x + "," + y + "," + z;
		putText(image2, txt, flow_corners[imgpts_back_index[i]], CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 3, CV_AA);
		cout << flow_corners[imgpts_back_index[i]] << " -> point cloud -> " << pointcloud[i] << endl;
	}

	imshow("hihi", image2);

	while (1) {
		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}



}

void refresh_features(Mat gray, vector<Point2f> &flow_c)
{
	//Mat mask(image_gray.size(), CV_8UC1, Scalar(255));
	int size_mask = 40;
	vector<Point2f> new_corners;
	Mat mask(gray.size(), CV_8UC1, Scalar(255));
	if (!flow_c.empty()) {
		for (int j = 0; j < flow_c.size(); j++) {
			circle(mask, flow_c[j], maskSize, 0, -1, 8, 0);
		}
	}
	goodFeaturesToTrack(gray, new_corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k_harris);
	flow_c.insert(flow_c.end(), new_corners.begin(), new_corners.end());

	imshow("Mask", mask);
}

void draw_opticalFlow(Point2f corners, Point2f flow_corners, Mat img, CvScalar line_color) {
	int line_thickness;				line_thickness = 3;

	CvPoint p, q;

	p.x = (int)corners.x;

	p.y = (int)corners.y;

	q.x = (int)flow_corners.x;

	q.y = (int)flow_corners.y;

	double angle;		angle = atan2((double)p.y - q.y, (double)p.x - q.x);

	double hypotenuse;	hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));

	q.x = (int)(p.x - 1 * hypotenuse * cos(angle));

	q.y = (int)(p.y - 1 * hypotenuse * sin(angle));

	line(img, p, q, line_color, line_thickness, CV_AA, 0);

	p.x = (int)(q.x + 5 * cos(angle + pi / 4));

	p.y = (int)(q.y + 5 * sin(angle + pi / 4));

	line(img, p, q, line_color, line_thickness, CV_AA, 0);

	p.x = (int)(q.x + 5 * cos(angle - pi / 4));

	p.y = (int)(q.y + 5 * sin(angle - pi / 4));

	line(img, p, q, line_color, line_thickness, CV_AA, 0);

}

bool CheckCoherentRotation(cv::Mat_<double>& R) {


	if (fabsf(determinant(R)) - 1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	return true;
}

Mat_<double> LinearLSTriangulation(
	Point3d u,//homogenous image point (u,v,1)
	Matx34d P,//camera 1 matrix
	Point3d u1,//homogenous image point in 2nd camera
	Matx34d P1//camera 2 matrix
	) { //build A matrix
		Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
			u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
			u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
			u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
			);
	//build B vector
	Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));
	//solve for X
	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);
	Mat_<double> X_(4, 1);
	X_(0) = X(0); X_(1) = X(1); X_(2) = X(2); X_(3) = 1.0;
	return X_;
}