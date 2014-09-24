/*
 * CAnnotation.cpp
 *
 *  Created on: Jun 13, 2014
 *      Author: test
 */

#include "CAnnotation.h"

CAnnotation::CAnnotation(string img_filename) : img_filename_(img_filename), exit_annotation_(false)
{
	img_ = imread(img_filename_);
	if(!img_.data)
	{
		std::cerr << "Annotation: File " << img_filename_ << " doesn't exist!" << std::endl;
	}
	else
	{
		rescale_view_factor_ = (float)IMG_VIEW_SIZE / (float)img_.cols;
	}
}

CAnnotation::~CAnnotation() {

}

CAnnotation::CAnnotation(const CAnnotation& other)
{
	this->cur_annotation_ = other.cur_annotation_;
	this->exit_annotation_ = other.exit_annotation_;
	this->img_filename_ = other.img_filename_;
	this->rescale_view_factor_ = other.rescale_view_factor_;
	this->sockets_ = other.sockets_;
	this->switches_ = other.switches_;
	this->rect_plane_ = other.rect_plane_;
}

bool CAnnotation::isImageLoaded()
{
	return img_.data ? true : false;
}

bool CAnnotation::annotate()
{
	// show image
	show();

	bool coninue_with_next_img = true;

	// get user input
	setMouseCallback("Annotate Image", &onMouse, this);
	char key;
	while(!exit_annotation_)
	{
		key = waitKey(100);

		switch(key)
		{
		case 'x': // on key 'x' -> save this is a socket
			if(cur_annotation_.size() == 4)
				sockets_.push_back(cur_annotation_);
			cur_annotation_.clear();
			show();
			break;
		case 's':  // on key 's' -> save this is a switch
			if(cur_annotation_.size() == 4)
				switches_.push_back(cur_annotation_);
			cur_annotation_.clear();
			show();
			break;
		case 'p':  // on key 'p' -> this is the rectification plane
			rect_plane_.clear();
			rect_plane_.insert(rect_plane_.begin(), cur_annotation_.begin(), cur_annotation_.end());
			cur_annotation_.clear();
			show();
			break;
		case 27: // on key ESC -> reset all annotations
			reset();
			show();
			break;
		case 32: // SPACE -> reset current annotation
			cur_annotation_.clear();
			show();
			break;
		case 'q': // q: quit annotation and quit program
			exit_annotation_ = true;
			coninue_with_next_img = false;
			break;
		case 'h': // print help text
			cout << "Keys are: x (socket), s (switch), SPACE (clear current), ESC (clear all), q (quit and save)" << endl;
			break;
		}
	}

	// save user input
	cout << "Annotate of current image finished. Got " << sockets_.size() << " sockets and " << switches_.size() << " switches." << endl;

	return coninue_with_next_img;
}

void CAnnotation::refine()
{
	vector<vector<Point> >::const_iterator an_it;
	for(an_it = sockets_.begin(); an_it != switches_.end(); ++an_it)
	{
		if(an_it == sockets_.end())
		{
			if(switches_.empty())
				break;

			an_it = switches_.begin();
		}
	}
}

void CAnnotation::show()
{

	//imshow("Annotate Image", img);

	Mat overlay_img = img_.clone();

	// draw annotations
	polylines(overlay_img, switches_, true, Scalar(0,0,255), 2);
	polylines(overlay_img, sockets_, true, Scalar(255,0,0), 2);
	if(!rect_plane_.empty())
		polylines(overlay_img, rect_plane_, true, Scalar(255,200,0), 2);

	overlay_img = overlay_img*0.5 + img_*0.5;

	bool close_cur_anno = cur_annotation_.size()==4;
	Scalar cur_anno_color(0, 255, close_cur_anno ? 0 : 255);
	if(!cur_annotation_.empty())
	{
		if(cur_annotation_.size() > 1)
			polylines(overlay_img, cur_annotation_, close_cur_anno, cur_anno_color, 2);
		else
			circle(overlay_img, cur_annotation_[0], 2, cur_anno_color, -1);
	}

	resize(overlay_img, overlay_img, Size(IMG_VIEW_SIZE, (int)((float)overlay_img.rows * rescale_view_factor_ )));
	imshow("Annotate Image", overlay_img);
}


void CAnnotation::zoomRegion(int x, int y)
{
	Mat region;
	if(img_.cols > IMG_ZOOM_REGION && img_.rows > IMG_ZOOM_REGION)
	{
		int x_real = x/rescale_view_factor_, y_real = y/rescale_view_factor_;
		Rect roi;
		roi.x = x_real-IMG_ZOOM_REGION/2;
		roi.y = y_real-IMG_ZOOM_REGION/2;
		roi.height = IMG_ZOOM_REGION;
		roi.width = IMG_ZOOM_REGION;

		roi.x = roi.x < 0 ? 0 : roi.x;
		roi.x = roi.x > img_.cols-roi.width ? img_.cols-roi.width : roi.x;
		roi.y = roi.y < 0 ? 0 : roi.y;
		roi.y = roi.y > img_.rows-roi.height ? img_.rows-roi.height : roi.y;

		Mat temp_img = img_.clone();
		polylines(temp_img, switches_, true, Scalar(0,0,255), 2);
		polylines(temp_img, sockets_, true, Scalar(255,0,0), 2);

		if(!cur_annotation_.empty())
		{
			bool close_cur_anno = cur_annotation_.size()==4;
			Scalar cur_anno_color(0, 255, close_cur_anno ? 0 : 255);

			if(cur_annotation_.size() > 1)
				polylines(temp_img, cur_annotation_, close_cur_anno, cur_anno_color, 2);
			else
				circle(temp_img, cur_annotation_[0], 2, cur_anno_color, -1);
		}


		(temp_img)(roi).copyTo(region);

		Point2i c(x_real - roi.x, y_real-roi.y);
		line(region, c-Point(0,IMG_ZOOM_REGION), c+Point(0,IMG_ZOOM_REGION), Scalar(0,0,255), 1);
		line(region, c-Point(IMG_ZOOM_REGION,0), c+Point(IMG_ZOOM_REGION,0), Scalar(0,0,255), 1);


	}
	else
	{
		region = img_;
	}

	imshow("zoom", region);
}


int CAnnotation::crop_socket_cnt_ = 0;
int CAnnotation::crop_switch_cnt_ = 0;
int CAnnotation::crop_neg_cnt_ = 0;

float rand_FloatRange(float low, float high)
{
	return ((high-low)*((float)rand()/RAND_MAX))+low;
}

void fillDestPoints(const string& type, vector<Point2f>& points, const float scale = 1.0)
{
	points.clear();
	if(type == "1_socket")
	{
		points.push_back(Point2f(SMP_FULL_SIZE_MM / 2.0 * PIXEL_PER_SQ_MM, (SMP_FULL_SIZE_MM-SOCKET_DIAMETER_MM*scale)/2.0 * PIXEL_PER_SQ_MM));
		points.push_back(Point2f(SMP_FULL_SIZE_MM / 2.0 * PIXEL_PER_SQ_MM, (SMP_FULL_SIZE_MM+SOCKET_DIAMETER_MM*scale)/2.0 * PIXEL_PER_SQ_MM));
		points.push_back(Point2f((SMP_FULL_SIZE_MM-SOCKET_DIAMETER_MM*scale)/2.0 * PIXEL_PER_SQ_MM, SMP_FULL_SIZE_MM / 2.0 * PIXEL_PER_SQ_MM));
		points.push_back(Point2f((SMP_FULL_SIZE_MM+SOCKET_DIAMETER_MM*scale)/2.0 * PIXEL_PER_SQ_MM, SMP_FULL_SIZE_MM / 2.0 * PIXEL_PER_SQ_MM));
	}
	else if(type == "2_switch")
	{
		points.push_back(Point2f((SMP_FULL_SIZE_MM - SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM,(SMP_FULL_SIZE_MM - SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM));
		points.push_back(Point2f((SMP_FULL_SIZE_MM - SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM,(SMP_FULL_SIZE_MM + SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM));
		points.push_back(Point2f((SMP_FULL_SIZE_MM + SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM,(SMP_FULL_SIZE_MM + SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM));
		points.push_back(Point2f((SMP_FULL_SIZE_MM + SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM,(SMP_FULL_SIZE_MM - SWITCH_SIZE_MM*scale)/2.0 * PIXEL_PER_SQ_MM));
	}
	else if(type == "plane")
	{
		points.push_back(Point2f(0,0));
		points.push_back(Point2f(0,+500));
		points.push_back(Point2f(+500,+500));
		points.push_back(Point2f(+500,0));
	}
}

void CAnnotation::saveAnnotatedImages(const string& prefix)
{
//	// read image
//	Mat img0 = imread(img_filename_);
//	if(!img0.data)
//	{
//		std::cerr << "Annotation: File doesn't exist. Something went terribly wrong!" << std::endl;
//		exit(-2);
//	}

	cout << "Cutting: Sockets: " << sockets_.size() << ", Switches: " << switches_.size() << endl;

	const float random_fluct_socket_px = 1.5;
	const float random_fluct_switch_px = 3.0;
	const float random_scale_range_socket_low = 1.0;
	const float random_scale_range_socket_high = 1.0;
	const float random_scale_range_switch_low = SWITCH_SIZE_MM_FROM / SWITCH_SIZE_MM;
	const float random_scale_range_switch_high = SWITCH_SIZE_MM_TO / SWITCH_SIZE_MM;

	vector<Point2f> dest_points;
	float random_fluct_px = random_fluct_socket_px;
	float random_scale_range_low = random_scale_range_socket_low;
	float random_scale_range_high = random_scale_range_socket_high;

	Mat plane_rect_transform_mat;
	if(!sockets_.empty() || !switches_.empty())
	{
		vector<Point2f> plane_dest_points;
		// plane rectification
		fillDestPoints("plane", plane_dest_points);
		vector<Point2f> rect_plane_float;
		for(vector<Point>::const_iterator pt_it = rect_plane_.begin(); pt_it != rect_plane_.end(); ++pt_it)
			rect_plane_float.push_back(Point2f(pt_it->x, pt_it->y));
		plane_rect_transform_mat = findHomography(rect_plane_float, plane_dest_points);
	}

	// mask where no negative samples should be taken
	Mat mask_map = Mat::zeros(img_.size(), CV_8UC1);

	// extract all positive samples (sockets and switches)
	string type = "1_socket";
	int* counter = &crop_socket_cnt_;
	vector<vector<Point> >::const_iterator an_it;
	for(an_it = sockets_.begin(); an_it != switches_.end(); ++an_it)
	{
		if(an_it == sockets_.end())
		{
			if(switches_.empty())
				break;

			an_it = switches_.begin();
			type = "2_switch";
			counter = &crop_switch_cnt_;
			random_fluct_px = random_fluct_switch_px;
			random_scale_range_low = random_scale_range_switch_low;
			random_scale_range_high = random_scale_range_switch_high;
		}

		float cur_random_fluct_px = 0.0; // for the first pos sample use the original annotations
		float cur_random_scale_low = 1.0;
		float cur_random_scale_high = 1.0;

		for(int pos_sample_cnt = 0; pos_sample_cnt < POS_SMP_PER_ANNO; ++pos_sample_cnt)
		{
			Point2f random_fluctuation;
			float cur_random_scale = rand_FloatRange(cur_random_scale_low, cur_random_scale_high);
			//cout << " " << pos_sample_cnt << " scale: " << cur_random_scale << endl;
			vector<Point2f> src_points;
			vector<Point>::const_iterator pt_it;
			for(pt_it = an_it->begin(); pt_it != an_it->end(); ++pt_it)
			{
				random_fluctuation.x = rand_FloatRange(-cur_random_fluct_px, +cur_random_fluct_px);
				random_fluctuation.y = rand_FloatRange(-cur_random_fluct_px, +cur_random_fluct_px);
				src_points.push_back(Point2f(pt_it->x, pt_it->y) + random_fluctuation);
			}

			const Point** corners_ptr = (const Point**)(&*an_it);
			int n_corners[] = { 4 };
			fillPoly(mask_map, corners_ptr, n_corners, 1, Scalar(255));

			// rectify annotated points according to plane
			perspectiveTransform(src_points, src_points, plane_rect_transform_mat);

			// find only affine transform to match with the rectified annotations
			fillDestPoints(type, dest_points, cur_random_scale);
			Mat affine_transform_mat = estimateRigidTransform(src_points, dest_points, true);

			Mat affine_transform_as_perspective = Mat::zeros(3,3,CV_64FC1);
			affine_transform_mat.copyTo(affine_transform_as_perspective(Rect(0,0,3,2)));
			affine_transform_as_perspective.at<double>(2, 2) = 1.0;

			//Mat homography = findHomography(src_points, dest_points);
			Mat cropped_img;
			Mat full_transformation = affine_transform_as_perspective*plane_rect_transform_mat;
			warpPerspective(img_, cropped_img, full_transformation, Size(SMP_FULL_SIZE_MM,SMP_FULL_SIZE_MM), INTER_CUBIC, BORDER_REPLICATE);
			stringstream img_name;
			img_name << prefix << type << setw(5) << setfill('0') << *counter << ".jpg";

			cout << "wrote file " << img_name.str() << endl;
			imwrite(img_name.str(), cropped_img);
			*counter = *counter + 1;

			cur_random_fluct_px = random_fluct_px; // add fluctuation (in pixel)
			cur_random_scale_low = random_scale_range_low; // add fluctuation in scale
			cur_random_scale_high = random_scale_range_high;
		}
	}

	//resize(mask_map, mask_map, img_.size());

	type = "0_neg";
	fillDestPoints("2_switch", dest_points);

	// assume a good negative sample in an image that contains annotations twice as big as in an pure negative image
	const float neg_sample_scale_factor = (sockets_.empty() && switches_.empty()) ? 1.0 : 2.0;
	// the size measure of a negative sample is 1/10 of the minimum dimension of the image
	const float neg_sample_size_px = min(img_.cols, img_.rows)*neg_sample_scale_factor/10.0;
	vector<Point2f> neg_sample_points;
	neg_sample_points.push_back(Point2f(-neg_sample_size_px/2.0, -neg_sample_size_px/2.0));
	neg_sample_points.push_back(Point2f(-neg_sample_size_px/2.0, +neg_sample_size_px/2.0));
	neg_sample_points.push_back(Point2f(+neg_sample_size_px/2.0, +neg_sample_size_px/2.0));
	neg_sample_points.push_back(Point2f(+neg_sample_size_px/2.0, -neg_sample_size_px/2.0));

	// get some negative examples from slightly manipulated and shifted annotations
	int neg_sample_cnt = 0, try_cnt = 0;
	while(neg_sample_cnt < NEG_SMP_PER_IMG && ++try_cnt < 10000)
	{
		vector<Point2f> src_points;
		vector<Point2f>::const_iterator pt_it;
		Point2f random_position;
		const float border = neg_sample_size_px/2.0;

		const float random_scale_range_low = 0.5, random_scale_range_high = 2.0;
		float random_scale = rand_FloatRange(random_scale_range_low, random_scale_range_high);
		random_position.x = rand_FloatRange(border*random_scale, img_.cols - border*random_scale);
		random_position.y = rand_FloatRange(border*random_scale, img_.rows - border*random_scale);
		//cout << "Random offset: " << random_offset;
		Point2f random_fluctuation;
//		vector<Point2i> temp_points;
		for(pt_it = neg_sample_points.begin(); pt_it != neg_sample_points.end(); ++pt_it)
		{
			const float random_fluct_range_low = -neg_sample_size_px/10.0, random_fluct_range_high = neg_sample_size_px/10.0;
			random_fluctuation.x = rand_FloatRange(random_fluct_range_low, random_fluct_range_high);
			random_fluctuation.y = rand_FloatRange(random_fluct_range_low, random_fluct_range_high);
			//cout << " random_fluc: " << random_fluctuation;
			Point2f new_point = (*pt_it +  random_fluctuation)*random_scale + random_position;
			//cout << " -> new point: " << new_point << " old was: " << *pt_it << endl;
			src_points.push_back(new_point);
//			temp_points.push_back(Point2i(new_point.x, new_point.y));
		}
		//if((sockets_.empty() && switches_.empty()))
//		{
//			Mat temp = img_.clone();
//			polylines(temp, temp_points, true, Scalar(255,0,0), 2);
//			PAUSE_AND_SHOW(temp)
//		}
		Mat homography = findHomography(src_points, dest_points);

		// see if it intersects with a positive sample or is outside the image (done with constant border handling)
		Mat cropped_mask_map;

		warpPerspective(mask_map, cropped_mask_map, homography, Size(SMP_FULL_SIZE_MM,SMP_FULL_SIZE_MM), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
		if(!countNonZero(cropped_mask_map))
		{
			Mat neg_sample;
			warpPerspective(img_, neg_sample, homography, Size(SMP_FULL_SIZE_MM,SMP_FULL_SIZE_MM));
			stringstream img_name;
			img_name << prefix << type << setw(6) << setfill('0') << crop_neg_cnt_ << ".jpg";

			cout << "wrote file " << img_name.str() << endl;
			imwrite(img_name.str(), neg_sample);
			neg_sample_cnt++;
			crop_neg_cnt_++;
		}
	}

}


void CAnnotation::addXMLString(tinyxml2::XMLDocument& doc)
{
	tinyxml2::XMLElement* xml_annotation = doc.NewElement("Annotation");
	xml_annotation->SetAttribute("image", this->img_filename_.c_str());

	tinyxml2::XMLElement* xml_plane = doc.NewElement("plane");
	vector<Point>::const_iterator pt_it;
	for(pt_it = rect_plane_.begin(); pt_it != rect_plane_.end(); ++pt_it)
	{
		tinyxml2::XMLElement* pt = doc.NewElement("point");

		stringstream str_x;
		str_x << pt_it->x;
		pt->SetAttribute("x", str_x.str().c_str());

		stringstream str_y;
		str_y << pt_it->y;
		pt->SetAttribute("y", str_y.str().c_str());

		xml_plane->InsertEndChild(pt);
	}
	xml_annotation->InsertEndChild(xml_plane);

	vector<vector<Point> >::const_iterator an_it;
	for(an_it = sockets_.begin(); an_it != sockets_.end(); ++an_it)
	{
		tinyxml2::XMLElement* xml_socket = doc.NewElement("socket");

		vector<Point>::const_iterator pt_it;
		for(pt_it = an_it->begin(); pt_it != an_it->end(); ++pt_it)
		{
			tinyxml2::XMLElement* pt = doc.NewElement("point");

			stringstream str_x;
			str_x << pt_it->x;
			pt->SetAttribute("x", str_x.str().c_str());

			stringstream str_y;
			str_y << pt_it->y;
			pt->SetAttribute("y", str_y.str().c_str());

			xml_socket->InsertEndChild(pt);
		}
		xml_annotation->InsertEndChild(xml_socket);
	}
	for(an_it = switches_.begin(); an_it != switches_.end(); ++an_it)
	{
		tinyxml2::XMLElement* xml_switch = doc.NewElement("switch");

		vector<Point>::const_iterator pt_it;
		for(pt_it = an_it->begin(); pt_it != an_it->end(); ++pt_it)
		{
			tinyxml2::XMLElement* pt = doc.NewElement("point");

			stringstream str_x;
			str_x << pt_it->x;
			pt->SetAttribute("x", str_x.str().c_str());

			stringstream str_y;
			str_y << pt_it->y;
			pt->SetAttribute("y", str_y.str().c_str());

			xml_switch->InsertEndChild(pt);
		}
		xml_annotation->InsertEndChild(xml_switch);
	}
	doc.InsertEndChild(xml_annotation);
}

void CAnnotation::loadFromXMLElement(XMLElement* xml_el)
{
	reset();

	XMLElement* plane_xml_el = xml_el->FirstChildElement("plane");
	if(plane_xml_el)
	{
		for(XMLElement* cur_point_xml = plane_xml_el->FirstChildElement("point"); cur_point_xml != NULL; cur_point_xml = cur_point_xml->NextSiblingElement("point"))
		{
			stringstream ss_x, ss_y;
			ss_x << cur_point_xml->Attribute("x");
			ss_y << cur_point_xml->Attribute("y");
			int x, y;
			ss_x >> x;
			ss_y >> y;
			rect_plane_.push_back(Point(x, y));
		}
	}

	XMLElement* cur_object_xml;
	for(cur_object_xml = xml_el->FirstChildElement("socket"); cur_object_xml != NULL; cur_object_xml = cur_object_xml->NextSiblingElement("socket"))
	{
		vector<Point> cur_points;
		XMLElement* cur_object_point_xml;
		for(cur_object_point_xml = cur_object_xml->FirstChildElement("point"); cur_object_point_xml != NULL; cur_object_point_xml = cur_object_point_xml->NextSiblingElement("point"))
		{
			stringstream ss_x, ss_y;
			ss_x << cur_object_point_xml->Attribute("x");
			ss_y << cur_object_point_xml->Attribute("y");
			int x, y;
			ss_x >> x;
			ss_y >> y;
			cur_points.push_back(Point(x, y));
		}
		sockets_.push_back(cur_points);
	}

	for(cur_object_xml = xml_el->FirstChildElement("switch"); cur_object_xml != NULL; cur_object_xml = cur_object_xml->NextSiblingElement("switch"))
	{
		vector<Point> cur_points;
		XMLElement* cur_object_point_xml;
		for(cur_object_point_xml = cur_object_xml->FirstChildElement("point"); cur_object_point_xml != NULL; cur_object_point_xml = cur_object_point_xml->NextSiblingElement("point"))
		{
			stringstream ss_x, ss_y;
			ss_x << cur_object_point_xml->Attribute("x");
			ss_y << cur_object_point_xml->Attribute("y");
			int x, y;
			ss_x >> x;
			ss_y >> y;
			cur_points.push_back(Point(x, y));
		}
		switches_.push_back(cur_points);
	}
}


void CAnnotation::mouseInputHandler(int event, int flag, int x, int y)
{

	if(event == CV_EVENT_RBUTTONDOWN) // finish annotation of the whole image on double click
	{
		exit_annotation_ = true;
	}
	else if(event == CV_EVENT_LBUTTONDOWN)
	{
		cur_annotation_.push_back(Point(x/rescale_view_factor_,y/rescale_view_factor_));
		show();
	}

}

void onMouse(int event, int x, int y, int flag, void* gui_ptr_arg)
{
	CAnnotation* anno_ptr = static_cast<CAnnotation*>(gui_ptr_arg);
	anno_ptr->zoomRegion(x, y);
    if(event != CV_EVENT_MOUSEMOVE)
    {
        anno_ptr->mouseInputHandler(event, flag, x, y);
    }
}

void CAnnotation::reset()
{
	rect_plane_.clear();
	sockets_.clear();
	switches_.clear();
	cur_annotation_.clear();
}
