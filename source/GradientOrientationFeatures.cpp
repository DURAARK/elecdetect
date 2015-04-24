/*
 * ElecDetec: GradientOrientationFeatures.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#include "GradientOrientationFeatures.h"


CGradientOrientationFeatures::CGradientOrientationFeatures() :
    n_tests_(GOF_N_TESTS),
    symm_ratio_(GOF_SYMM_RATIO),
    pair_ratio_(GOF_PAIR_RATIO),
    min_rect_sz_(GOF_MIN_RECT_SZ),
    max_rect_sz_(_PATCH_WINDOW_SIZE_),
    mag_threshold_(GOF_MAG_THRESHOLD)
{
    // the pair-wise test is a cosine-similarity measure. Only the orientation is considered, i.e. if
    // the image-pixels are radiants, 0 and pi are considered as similar.
    // As orientation of one pixel, the gradient with the highest magnitude of the 3 color channel is
	// assigned
	initWavelets();
}

CGradientOrientationFeatures::~CGradientOrientationFeatures()
{

}

void CGradientOrientationFeatures::initWavelets()
{
	// generate test-boxes (relative coordinates within the image patch)
    wavelet_pairs_.clear();

    bool make_next_single_sym = false;
	for(int i = 0; i < n_tests_; ++i)
	{
		Rect rect1;

        // if last Rect was single and symmteric, make the current single and symmetric to the last one
        if(make_next_single_sym)
        {
            Rect last_rect = wavelet_pairs_.back().first;
            // Percentage of the types of the symmetric arranged pairs:
            // Horizontal-symmetric, Vertical-symmetric, Center-symmetric
            const float percent_h_sym = 0.3;
            const float percent_v_sym = 0.3; // the rest is center-symmetric

            float sym_type = randRange<float>(0,1);

            rect1.height = last_rect.height;
            rect1.width = last_rect.width;
            if(sym_type < percent_h_sym) // h-symmetric
            {
                rect1.x = _PATCH_WINDOW_SIZE_ - last_rect.x - last_rect.width; // see notes for derivation
                rect1.y = last_rect.y;
            }
            else if(sym_type < percent_h_sym+percent_v_sym) // v-symmetric
            {
                rect1.x = last_rect.x;
                rect1.y = _PATCH_WINDOW_SIZE_ - last_rect.y - last_rect.height;
            }
            else // c-symmetric
            {
                rect1.x = _PATCH_WINDOW_SIZE_ - last_rect.x - last_rect.width;
                rect1.y = _PATCH_WINDOW_SIZE_ - last_rect.y - last_rect.height;
            }

            wavelet_pairs_.push_back(RectPair(rect1, Rect(0,0,0,0)));
            make_next_single_sym = false;
            continue;
        }

        rect1.width = randRange<int>(min_rect_sz_, max_rect_sz_);
        rect1.height = randRange<int>(min_rect_sz_, max_rect_sz_);
        rect1.x = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect1.width-1);
        rect1.y = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect1.height-1);

        Rect rect2(0,0,0,0);
        if(pair_ratio_ > randRange<float>(0,1)) // should it be a pair-wise test?
		{
            if(symm_ratio_ > randRange<float>(0,1)) // should the pairs be symmetric?
			{
				// Percentage of the types of the symmetric arranged pairs:
				// Horizontal-symmetric, Vertical-symmetric, Center-symmetric
				const float percent_h_sym = 0.3;
				const float percent_v_sym = 0.3; // the rest is center-symmetric

				float sym_type = randRange<float>(0,1);

				rect2.height = rect1.height;
				rect2.width = rect1.width;
				if(sym_type < percent_h_sym) // h-symmetric
				{
                    rect2.x = _PATCH_WINDOW_SIZE_ - rect1.x - rect1.width; // see notes for derivation
					rect2.y = rect1.y;
				}
				else if(sym_type < percent_h_sym+percent_v_sym) // v-symmetric
				{
					rect2.x = rect1.x;
                    rect2.y = _PATCH_WINDOW_SIZE_ - rect1.y - rect1.height;
				}
				else // c-symmetric
				{
                    rect2.x = _PATCH_WINDOW_SIZE_ - rect1.x - rect1.width;
                    rect2.y = _PATCH_WINDOW_SIZE_ - rect1.y - rect1.height;
				}
			}
			else
			{
                rect2.width = randRange<int>(min_rect_sz_, max_rect_sz_);
                rect2.height = randRange<int>(min_rect_sz_, max_rect_sz_);
                rect2.x = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect2.width-1);
                rect2.y = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect2.height-1);
			}
		}
        else // no pair. should it be a single symmetric rect?
        {
            if(symm_ratio_ > randRange<float>(0,1))
            {
                make_next_single_sym = true;
            }
        }
		wavelet_pairs_.push_back(RectPair(rect1, rect2));
	}

}

void CGradientOrientationFeatures::extractFeatureVector(const Mat& input_img, vector<float>& output_vec)
{
    output_vec.clear();

	Mat color_channels[3];
    split(input_img, color_channels);

	// Step 1: get for each image pixel the gradient for each color channel
	Mat channel_mag_resp[3];
	Mat d_x[3], d_y[3];
	for(int ccnt = 0; ccnt < 3; ++ccnt)
	{
		Mat cur_channel = color_channels[ccnt];

		// Blur
		GaussianBlur(color_channels[ccnt], cur_channel, Size(3,3), 0, 0, BORDER_DEFAULT);
		// Gradient X
        Sobel(cur_channel, d_x[ccnt], CV_32F, 1, 0, 3);
		// Gradient Y
        Sobel(cur_channel, d_y[ccnt], CV_32F, 0, 1, 3);

		// superposition of the absolute of both directions
		magnitude(d_x[ccnt], d_y[ccnt], channel_mag_resp[ccnt]);
	}

	// Step 2: for each pixel, find the largest gradient of the color channels
    Mat thresholded_dominant_mag = Mat::zeros(input_img.size(), CV_32FC1);
    Mat dominant_orientations_unit_vec = Mat::zeros(input_img.size(), CV_32FC2);
    for(int row_cnt = 0; row_cnt < dominant_orientations_unit_vec.rows; ++row_cnt)
	{
        for(int col_cnt = 0; col_cnt < dominant_orientations_unit_vec.cols; ++col_cnt)
		{
			// for each point in the image
            const Point2i pt(col_cnt, row_cnt);
			int dominant_ch_idx = -1;

			// search for RGB channel with the largest magnitude > mag_threshold
			float max_val = mag_threshold_;
			for(uchar ccnt = 0; ccnt < 3; ++ccnt)
			{
				if(channel_mag_resp[ccnt].at<float>(pt) > max_val)
				{
					max_val = channel_mag_resp[ccnt].at<float>(pt);
					dominant_ch_idx = ccnt;
				}
			}
			if(dominant_ch_idx != -1)
			{
				thresholded_dominant_mag.at<float>(pt) = max_val;
				float unit_vec_x = d_x[dominant_ch_idx].at<float>(pt)/max_val;
				float unit_vec_y = d_y[dominant_ch_idx].at<float>(pt)/max_val;
				dominant_orientations_unit_vec.at<Vec2f>(pt) = Vec2f(unit_vec_x, unit_vec_y); // only consider the orientation (0..pi)
				//cout << sqrt(unit_vec_x*unit_vec_x + unit_vec_y*unit_vec_y);
			}
		}
	}

//    Mat vis_x;
//    d_x[0].convertTo(vis_x, CV_8UC1);
//    PAUSE_AND_SHOW(vis_x)

    // abs dot product images
    const Vec2f vec0deg(1,0);
    const Vec2f vec45deg(0.707107, 0.707107);
    const Vec2f vec90deg(0,1);
    const Vec2f vec135deg(-0.707107, 0.707107);

    // calculate dot products
    Mat dotprods = Mat::zeros(input_img.size(), CV_32FC4);
    for(int row_cnt = 0; row_cnt < dominant_orientations_unit_vec.rows; ++row_cnt)
    {
        for(int col_cnt = 0; col_cnt < dominant_orientations_unit_vec.cols; ++col_cnt)
        {
            Point2i pt(col_cnt, row_cnt);
            if(thresholded_dominant_mag.at<float>(pt) != 0)
            {
                // only consider the orientation (0..pi) by calculating just the absoulte dotproducts
                const Vec4f dotps( fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec0deg)),
                                   fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec45deg)),
                                   fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec90deg)),
                                   fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec135deg))
                                   );

                dotprods.at<Vec4f>(pt) = dotps;
            }
        }
    }


    // VIS -  Dominant orientation visualization:
//    Mat vis_img = Mat::zeros(working_data.data().size(), CV_8UC3);
//    for(int row_cnt = 0; row_cnt < dominant_orientations_unit_vec.rows; ++row_cnt)
//    {
//        for(int col_cnt = 0; col_cnt < dominant_orientations_unit_vec.cols; ++col_cnt)
//        {
//            Point2i pt(col_cnt, row_cnt);
//            if(thresholded_dominant_mag.at<float>(pt) != 0)
//            {
//                const float i = 1;//thresholded_dominant_mag.at<float>(pt)/255;
//                const uchar r = fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec0deg))*255;
//                const uchar g = fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec90deg))*255;
//                const uchar b = fabs(dominant_orientations_unit_vec.at<Vec2f>(pt).dot(vec45deg))*255;
//                vis_img.at<Vec3b>(pt) = Vec3b(b*i, g*i, r*i);
//            }
//        }
//    }
//    imwrite("orientations.png", vis_img);
//    PAUSE_AND_SHOW(vis_img);
//    vis_img = Mat::zeros(working_data.data().size(), CV_8UC1);
//    for(int row_cnt = 0; row_cnt < dominant_orientations_unit_vec.rows; ++row_cnt)
//    {
//        for(int col_cnt = 0; col_cnt < dominant_orientations_unit_vec.cols; ++col_cnt)
//        {
//            Point2i pt(col_cnt, row_cnt);
//            if(thresholded_dominant_mag.at<float>(pt) != 0)
//            {
//                vis_img.at<uchar>(pt) = thresholded_dominant_mag.at<float>(pt);
//            }
//        }
//    }
//    PAUSE_AND_SHOW(vis_img);
//	int grad_vis_cnt = 0;
//	for(int row_cnt = 0; row_cnt < dominant_orientations_unit_vec.rows; ++row_cnt)
//	{
//		for(int col_cnt = 0; col_cnt < dominant_orientations_unit_vec.cols; ++col_cnt)
//		{
//			Point2i pt(col_cnt, row_cnt);
//			if(grad_vis_cnt++ > 10 && thresholded_dominant_mag.at<float>(pt) != 0)
//			{
//				const float l = thresholded_dominant_mag.at<float>(pt)/50;
//				Point2i pt2 = pt + Point2i(-dominant_orientations_unit_vec.at<Vec2f>(pt).val[1]*l, dominant_orientations_unit_vec.at<Vec2f>(pt).val[0]*l);
//				line(vis_img, pt, pt2, Scalar(0,0,255), 1);
//				grad_vis_cnt = 0;
//			}
//		}
//	}
//	PAUSE_AND_SHOW(vis_img);
    // VIS


    // Calculate orientation Haar features:
    output_vec.reserve(4*wavelet_pairs_.size());

    Mat integral_dotprods;
    integral(dotprods, integral_dotprods, CV_32F);

    for(vector<RectPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
    {
        const Point tl1 = pair_it->first.tl();
        const Point tr1(pair_it->first.x + pair_it->first.width, pair_it->first.y);
        const Point br1 = pair_it->first.br();
        const Point bl1(pair_it->first.x, pair_it->first.y + pair_it->first.height);

        Vec4f rect_hist = integral_dotprods.at<Vec4f>(br1) - integral_dotprods.at<Vec4f>(bl1) - integral_dotprods.at<Vec4f>(tr1) + integral_dotprods.at<Vec4f>(tl1);
        const float hist1_sum = rect_hist[0] + rect_hist[1] + rect_hist[2] + rect_hist[3];
        if(hist1_sum > 1.0)
            rect_hist = rect_hist / hist1_sum;
        else
            rect_hist = Vec4f(0,0,0,0);

        if(pair_it->second.area()) // compare two rects
        {
            const Point tl2 = pair_it->second.tl();
            const Point tr2(pair_it->second.x + pair_it->second.width, pair_it->second.y);
            const Point br2 = pair_it->second.br();
            const Point bl2(pair_it->second.x, pair_it->second.y + pair_it->second.height);

            const Vec4f rect2_hist = integral_dotprods.at<Vec4f>(br2) - integral_dotprods.at<Vec4f>(bl2) - integral_dotprods.at<Vec4f>(tr2) + integral_dotprods.at<Vec4f>(tl2);
            const float hist2_sum = rect2_hist[0] + rect2_hist[1] + rect2_hist[2] + rect2_hist[3];

            if(hist2_sum > 1.0)
                rect_hist = rect_hist - (rect2_hist / hist2_sum);
        }

        output_vec.push_back(rect_hist[0]);
        output_vec.push_back(rect_hist[1]);
        output_vec.push_back(rect_hist[2]);
        output_vec.push_back(rect_hist[3]);
    }
}

void CGradientOrientationFeatures::save(FileStorage& fs) const
{
    fs << "n-tests" << n_tests_;
    fs << "symm-percent" << symm_ratio_;
    fs << "pair-percent" << pair_ratio_;
    fs << "min-rect-sz" << min_rect_sz_;
    fs << "max-rect-sz" << max_rect_sz_;
    fs << "mag-threshold" << mag_threshold_;
    fs << "rect-pairs" << "[:";
	for(vector<RectPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
    {
        fs << pair_it->first;
        fs << pair_it->second;
    }
	fs << "]";
}

void CGradientOrientationFeatures::load(FileNode& _node)
{
	wavelet_pairs_.clear();

    _node["n-tests"] >> n_tests_;
    _node["symm-percent"] >> symm_ratio_;
    _node["pair-percent"] >> pair_ratio_;
    _node["min-rect-sz"] >> min_rect_sz_;
    _node["max-rect-sz"] >> max_rect_sz_;
    _node["mag-threshold"] >> mag_threshold_;

    FileNode rect_pairs_node = _node["rect-pairs"];
	for(FileNodeIterator pair_node_it = rect_pairs_node.begin(); pair_node_it != rect_pairs_node.end(); ++pair_node_it)
	{
		Rect rect1, rect2;
		(*pair_node_it) >> rect1;
		pair_node_it++;
		(*pair_node_it) >> rect2;
		wavelet_pairs_.push_back(RectPair(rect1, rect2));
	}
}
