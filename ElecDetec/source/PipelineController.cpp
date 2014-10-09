/*
 * PipelineController.cpp
 *
 *  Created on: Jul 3, 2014
 *      Author: test
 */

#include "PipelineController.h"


CPipelineController::CPipelineController() : n_object_classes_(0)
{
}


CPipelineController::~CPipelineController()
{
	deletePipe();
}


void CPipelineController::deletePipe()
{
	// clean up modules
	for (vector<CVisionModule*>::const_iterator mod_it = all_modules_.begin(); mod_it != all_modules_.end(); ++mod_it)
	{
		delete *mod_it;
	}
	all_modules_.clear();
}


void CPipelineController::initializeFromParameters() throw (PipeConfigExecption)
{
	deletePipe(); // delete Pipe if it's already set up

	// Set up Feature Channels for the simple structure
	bool is_root = true;
	vector<CVisionModule*> end_modules_of_channels;
	vector<vector<string> >::const_iterator ch_str_it;
	for(ch_str_it = params_.vec_vec_channels_.begin(); ch_str_it != params_.vec_vec_channels_.end(); ++ch_str_it)
	{
		is_root = true;
		CVisionModule* ancestor_mod = NULL;
		vector<string>::const_iterator mod_str_it;
		for(mod_str_it = ch_str_it->begin(); mod_str_it != ch_str_it->end(); ++mod_str_it)
		{
			CVisionModule* cur_mod = NULL;

			string mod_id;
			string mod_param;
			size_t param_pos = mod_str_it->find(MODULE_PARAM_PRELUDE);
			if(param_pos != string::npos)
			{
				if(mod_str_it->find(MODULE_PARAM_ENDING) != mod_str_it->size()-1)
				{
					throw(PipeConfigExecption("Parameter sequence ending character not found at the end.")); // Param-Ending is not at the end
				}
				mod_param = mod_str_it->substr(param_pos+1, mod_str_it->size()-param_pos-2);
			}

			mod_id = mod_str_it->substr(0,param_pos);

			//cout << "Mod-ID: " << mod_id << " Params: " << mod_param << endl;

			// Pre-processing Methods
			if(mod_id == ID_CANNY)
				cur_mod = new CBinaryContours(is_root, mod_param);
			if(mod_id == ID_GRADIENT)
				cur_mod = new CGradientImage(is_root, mod_param);
			if(mod_id == ID_DISTTR)
				cur_mod = new CDistanceTransform(is_root, mod_param);
			if(mod_id == ID_QGRAD)
				cur_mod = new CQuantizedGradient(is_root, mod_param);
			if(mod_id == ID_ORI)
				cur_mod = new COrientationFilter(is_root, mod_param);
			if(mod_id == ID_CCHAN)
				cur_mod = new CColorChannel(is_root, mod_param);

			// Feature Extractor Methods
			if(mod_id == ID_HOG)
				cur_mod = new CHog(is_root, mod_param);
			if(mod_id == ID_BRIEF)
				cur_mod = new COwnBrief(is_root, mod_param);
			if(mod_id == ID_HAAR_FEAT)
				cur_mod = new CHaarWavelets(is_root, mod_param);
			if(mod_id == ID_GOF)
				cur_mod = new CGradientOrientationFeatures(is_root, mod_param);

			// Subspace methods
			if(mod_id == ID_PCA)
				cur_mod = new CPCA(is_root, mod_param);

			// Classifiers: TODO: re-implementation of classifiers needed (prediction probability)
			if(mod_id == ID_SVM)
				cur_mod = new CSVM(is_root, mod_param);
			if(mod_id == ID_LIN_SVM)
				cur_mod = new CLinSVM(is_root, mod_param);
			if(mod_id == ID_RF)
				cur_mod = new CRandomForest(is_root, mod_param);

			if(!cur_mod)
				throw(PipeConfigExecption()); // unknown ID given!!!

			if(ancestor_mod)
				ancestor_mod->setSuccessor(cur_mod);

			is_root = false;
			ancestor_mod = cur_mod;

			all_modules_.push_back(cur_mod);
		}
		end_modules_of_channels.push_back(ancestor_mod);
	}

	// Create final classifier
	CVisionModule* final_classifier = NULL;

	string mod_id;
	string mod_param;
	size_t param_pos = params_.str_classifier_.find(MODULE_PARAM_DELIMITER);
	if(param_pos != string::npos)
		mod_param = params_.str_classifier_.substr(param_pos+1);
	mod_id = params_.str_classifier_.substr(0,param_pos);

	if(mod_id == ID_SVM)
		final_classifier = new CSVM(is_root, mod_param);
	if(mod_id == ID_LIN_SVM)
		final_classifier = new CLinSVM(is_root, mod_param);
	if(mod_id == ID_RF)
		final_classifier = new CRandomForest(is_root, mod_param);

	if(!final_classifier)
		throw(PipeConfigExecption("Final classifier not specified or unknown ID given.")); // unknown ID given!!!

	// set successor of the last modules of the channels
	for (vector<CVisionModule*>::const_iterator mod_it = end_modules_of_channels.begin(); mod_it != end_modules_of_channels.end(); ++mod_it)
	{
		(*mod_it)->setSuccessor(final_classifier);
	}

	all_modules_.push_back(final_classifier);

	// set ID for each module (required for identifying in configuration file)
	int module_id = 0;
	for (vector<CVisionModule*>::const_iterator mod_it = all_modules_.begin(); mod_it != all_modules_.end(); ++mod_it)
	{
		(*mod_it)->setModuleID(module_id++);
	}


	printConfig();

	waitKey(0);
}

void CPipelineController::test(const Mat& input_img, vector<vector<Rect> >& bb_results)
{
	Mat mat_detect_labels = Mat::zeros(input_img.size(), CV_8UC1);  // discrete labels
	Mat mat_detect_prop = Mat::zeros(input_img.size(), CV_32FC1);   // probability map of the detection results
	Mat mat_to_analyze = Mat::zeros(input_img.size(), CV_8UC1);     // Binary image that codes which pixels (BB-top-left points)
	                                                                // are going to be analyzed

	cout << "Analyzing image...        " << flush;
	float progress = 0.0;

	// Scan Grid 'sg'
	Rect win_roi(0, 0, SWIN_SIZE, SWIN_SIZE);
	vector<int> sg_x, sg_y;
	linspace<int>(sg_x, 0, input_img.cols-SWIN_SIZE-1, ceil(((float)input_img.cols-SWIN_SIZE)/(float)OPENING_SIZE)+2);
	linspace<int>(sg_y, 0, input_img.rows-SWIN_SIZE-1, ceil(((float)input_img.rows-SWIN_SIZE)/(float)OPENING_SIZE)+2);
//	cout << "image is " << input_img.cols << " x " << input_img.rows << endl;
//	cout << "linspace x n_elements: " << sg_x.size() << " and y: " << sg_y.size() << endl;

	vector<Point> tl_scan_points;
	for(vector<int>::const_iterator y_it = sg_y.begin(); y_it != sg_y.end(); ++y_it)
	{
		for(vector<int>::const_iterator x_it = sg_x.begin(); x_it != sg_x.end(); ++x_it)
		{
			tl_scan_points.push_back(Point(*x_it, *y_it));
			mat_to_analyze.at<uchar>(Point(*x_it, *y_it)) = true;
		}
	}

//	cout << "x: ";
//	for(int i = 0; i < sg_x.size(); ++i)
//		cout << " " << sg_x[i];
//	cout << endl;
//
//	cout << "y: ";
//	for(int i = 0; i < sg_y.size(); ++i)
//		cout << " " << sg_y[i];
//	cout << endl;
//

	int progress_refresh_cnt = 0;
	for(unsigned int scan_pt_cnt = 0; scan_pt_cnt < tl_scan_points.size(); ++scan_pt_cnt)
	{
		win_roi.x = tl_scan_points[scan_pt_cnt].x;
		win_roi.y = tl_scan_points[scan_pt_cnt].y;

		if(progress_refresh_cnt++ == 100)
		{
			progress = 100.0f*(float)scan_pt_cnt/tl_scan_points.size();

			for(int backspace_cnt = 0; backspace_cnt < 6; ++backspace_cnt)
				cout << '\b';

			cout << setfill('0');
			cout << setw(5) << setiosflags(ios::fixed) << setprecision(2) << progress << "%" << flush;
			progress_refresh_cnt = 0;
		}

		// Execute each Pipeline and save results
		CVisionModule* cmp = all_modules_[0]; // Current Module Ptr: grab a root module
		CVisionModule* lmp = NULL;            // Last Vision Module Pointer: used to identify correct buffer and converter inside a module
		CVisionData* current_data_ptr = new CVisionData(input_img(win_roi), DATA_TYPE_IMAGE);
		while(cmp)
		{
			cmp->bufferData(current_data_ptr, lmp);
			delete current_data_ptr; // cleanup old data should be OK, since it is buffered in the module and cv::Mat data is not deleted
			current_data_ptr = NULL;
			if(cmp->isTrained())
			{
				CVisionModule* missing_ancestor = cmp->getAncestorModuleFromWhichNoDataIsBuffered();
				if(!missing_ancestor)
				{
					current_data_ptr = cmp->exec(); // also clears Data Buffer of current Module
					lmp = cmp;
					cmp = cmp->getSuccessor();

					continue;
				}
				else // if there is data missing: crawl the structure upwards the path to get data
				{
					while(missing_ancestor)
					{
						cmp = missing_ancestor;
						missing_ancestor = cmp->getAncestorModuleFromWhichNoDataIsBuffered();
						lmp = missing_ancestor; // is NULL when reached a root module
					}
					delete current_data_ptr;
					current_data_ptr = new CVisionData(input_img(win_roi), DATA_TYPE_IMAGE);
					continue;
				}
			}
			else
			{
				cerr << "Testing untrained modules?" << endl;
				exit(-1);
			}
		}

		// Classify with final classifier
		if(!SIGNATURE_IS_WSCALAR(current_data_ptr->getSignature()))
		{
			cerr << "Final Module has produced a not weighed scalar as result" << endl;
			exit(-1);
		}

		uchar result_class = static_cast<uchar>(current_data_ptr->data().at<float>(0,0));

		// if an object was found on this position
		if(result_class)
		{
			Point2i win_center(win_roi.x + win_roi.width/2, win_roi.y + win_roi.height/2);
			mat_detect_labels.at<uchar>(win_center) = result_class;
			mat_detect_prop.at<float>(win_center) = current_data_ptr->data().at<float>(0,1);

			// add surrounding TOP LEFT points to analyze
			Point from(win_roi.x - OPENING_SIZE, win_roi.y - OPENING_SIZE);
			from.x = from.x >= 0 ? from.x : 0;
			from.y = from.y >= 0 ? from.y : 0;

			Point to(win_roi.x + OPENING_SIZE, win_roi.y + OPENING_SIZE);
			to.x = to.x <= input_img.cols - SWIN_SIZE - 1 ? to.x : input_img.cols - SWIN_SIZE - 1;
			to.y = to.y <= input_img.rows - SWIN_SIZE - 1 ? to.y : input_img.rows - SWIN_SIZE - 1;

			for(int x = from.x; x <= to.x; ++x)
			{
				for(int y = from.y; y <= to.y; ++y)
				{
					Point cur_pt(x,y);
					if(!mat_to_analyze.at<uchar>(cur_pt)) // add the points just once
					{
						tl_scan_points.push_back(cur_pt);
						mat_to_analyze.at<uchar>(cur_pt) = true;
					}
				}
			}

		}

		// cleanup
		if(current_data_ptr)
			delete current_data_ptr;
		current_data_ptr = NULL;
	}

//	int step_xy = 1;
//	for(win_roi.y = 0; win_roi.y < input_img.rows-SWIN_SIZE; win_roi.y += step_xy)
//	{
//		progress = 100.0f*win_roi.y/(input_img.rows-SWIN_SIZE);
//		for(int backspace_cnt = 0; backspace_cnt < 6; ++backspace_cnt)
//			cout << '\b';
//		cout << setfill('0');
//		cout << setw(5) << setiosflags(ios::fixed) << setprecision(2) << progress << "%" << flush;
//
////#pragma omp parallel
//		for(win_roi.x = 0; win_roi.x < input_img.cols-SWIN_SIZE; win_roi.x += step_xy)
//		{
//			//CVisionData window(input_img(win_roi), DATA_TYPE_IMAGE);
//			//			window->show();
//			//			waitKey(3);
//
//
//
//		}
//	}

	//cout << "non zero: labels: " << countNonZero(mat_detect_labels) << " probs: " << countNonZero(mat_detect_prop) << endl;
	//PAUSE_AND_SHOW(mat_detect_prop)


	// Postprocessing of detection results
	postProcessResults(mat_detect_labels, mat_detect_prop, bb_results);

	for(int backspace_cnt = 0; backspace_cnt < 6; ++backspace_cnt)
		cout << '\b';
	cout << "finished." << endl;


}

void CPipelineController::postProcessResults(const Mat& labels0, const Mat& probability_map0, vector<vector<Rect> >& results)
{
	assert(labels0.type() == CV_8UC1 && probability_map0.type() == CV_32FC1 && labels0.size() == probability_map0.size());

	// some fiixed parameters for post-processing
	const float ms_kernel_radius = (float)SWIN_SIZE/10.0;
	const float max_object_overlap = 0.4;

	// copy input for editing
	Mat labels = labels0.clone();
	Mat probability_map = probability_map0.clone();

	// if detection result is binary (non-weighted results) estimate by Gauss-kernel
	if(countNonZero(probability_map) == countNonZero(probability_map == 1))
	{
		labels = Mat::zeros(labels0.size(), CV_8UC1);
		probability_map = Mat::zeros(labels0.size(), CV_32FC1);

		// first do opening per class
		for(int label_it = 1; label_it <= n_object_classes_; ++label_it)
		{
			Mat closed_mask = Mat::zeros(labels0.size(), CV_8UC1);
			morphologyEx(labels0 == label_it, closed_mask, MORPH_OPEN,
					getStructuringElement(MORPH_RECT, Size(OPENING_SIZE, OPENING_SIZE) ) ); // Rect due to SW stepsize = OPENING_SIZE

			labels0.copyTo(labels, closed_mask); // aware that higher labels may replace lower labels

			// and Gauß-filter to estimate probability
			Mat closed_mask_float = Mat::zeros(labels0.size(), CV_32FC1);
			closed_mask.convertTo(closed_mask_float, CV_32FC1, 1.0/255);
			int ksize = 2*static_cast<int>(SWIN_SIZE/2)+1;
			GaussianBlur(closed_mask_float, closed_mask_float, Size(ksize, ksize), (float)SWIN_SIZE/15.0);

			closed_mask_float.copyTo(probability_map, closed_mask); // aware that higher labels may replace lower labels
		}

#ifdef VERBOSE
		Mat vis_img;

		vis_img = Mat::zeros(labels0.size(), CV_8UC3);
		vis_img.setTo(Scalar(255,0,0), labels0 == 1);
		vis_img.setTo(Scalar(0,0,255), labels0 == 2);

		imshow("Original Labels", vis_img);
		imwrite("original_labels.jpg", vis_img);

		vis_img = Mat::zeros(labels0.size(), CV_8UC3);
		vis_img.setTo(Scalar(255,0,0), labels == 1);
		vis_img.setTo(Scalar(0,0,255), labels == 2);

		imshow("Closed Labels", vis_img);
		imwrite("closed_labels.jpg", vis_img);

		// - visualization ------------------------
		Mat prop_map_vis = Mat::zeros(probability_map.size(), CV_32FC1);
		normalize(probability_map, prop_map_vis, 0, 1.0, NORM_MINMAX);
		vis_img = Mat::zeros(labels0.size(), CV_8UC3);
		prop_map_vis.convertTo(vis_img, CV_8UC3, 255.0);
		imshow("Probabilities of Labels", vis_img);
		imwrite("prop_map.jpg", vis_img);
		// ----------------------------------------
#endif

	}


	// Perform meanshift clustering
	vector<CLabeledWeightedRect> objects;

	// for each point of each label
	for(int label_it = 1; label_it <= n_object_classes_; ++label_it)
	{
		vector<CLabeledWeightedRect> objects_of_cur_label;

		// initialize free object center map
		const int free = -1;
		Mat object_id_map(probability_map.size(), CV_32SC1, free);
		int object_cnt = 0;
		int object_id = 0;

		Mat label_mask = labels == label_it;
		Mat nonZeroIdx;
		findNonZero(label_mask, nonZeroIdx);
		for(unsigned int pt_it = 0; pt_it < nonZeroIdx.total(); ++pt_it)
		{
			Point seed = nonZeroIdx.at<Point>(pt_it);

			Point2f cur_state = seed;
			const int max_iterations = 20;
			const float epsilon = 0.05;

			float last_msvec_length = 10.0;
			int ms_iter = 0;
			for(ms_iter = 0; ms_iter < max_iterations && last_msvec_length > epsilon; ++ms_iter)
			{
				// gather points in range of the Euclidean distance
				vector<Point> points_in_range;
				float weight_sum = 0;
				for(unsigned int cmp_pt_it = 0; cmp_pt_it != nonZeroIdx.total() - 1; ++cmp_pt_it)
				{
					Point2f dist(cur_state.x - nonZeroIdx.at<Point>(cmp_pt_it).x,
							     cur_state.y - nonZeroIdx.at<Point>(cmp_pt_it).y);
					if(norm(dist) <= ms_kernel_radius)
					{
						points_in_range.push_back(nonZeroIdx.at<Point>(cmp_pt_it));
						weight_sum += probability_map.at<float>(nonZeroIdx.at<Point>(cmp_pt_it));
					}
				}
				// calculate weighted mean of the points within the kernel
				Point2f weighted_mean(0.0, 0.0);
				vector<Point>::const_iterator pt_in_range_it;
				for(pt_in_range_it = points_in_range.begin(); pt_in_range_it != points_in_range.end(); ++pt_in_range_it)
				{
					weighted_mean += Point2f(pt_in_range_it->x * probability_map.at<float>(*pt_in_range_it) / weight_sum,
							                 pt_in_range_it->y * probability_map.at<float>(*pt_in_range_it) / weight_sum);
				}

				last_msvec_length = norm(cur_state - weighted_mean);
				cur_state = weighted_mean;
			}
			//cout << "cluster center: " << cur_state << " (iterations: " << ms_iter << ")" << endl << flush;

			// Mean-Shift result is now cur_state
			// ------------  collect similar cluster centers ----------------------------------------

			// yes, we are aware of rounding errors
			Point2i ms_result(floor(cur_state.x+0.5), floor(cur_state.y+0.5));
			ms_result.x = ms_result.x < 0 ? 0 : ms_result.x;
			ms_result.y = ms_result.y < 0 ? 0 : ms_result.y;
			ms_result.x = ms_result.x > object_id_map.cols-1 ? object_id_map.cols-1 : ms_result.x;
			ms_result.y = ms_result.y > object_id_map.rows-1 ? object_id_map.rows-1 : ms_result.y;

			// draw at the found cluster center a circle with an id-value to mark the ms result
			if(object_id_map.at<int>(ms_result) == free)
			{
				// if we found a new cluster center
				object_id = object_cnt;
				object_cnt++;
				// and store the ms_result as new object center
				Rect obj_bb(ms_result.x - SWIN_SIZE/2, ms_result.y - SWIN_SIZE/2, SWIN_SIZE, SWIN_SIZE);
				objects_of_cur_label.push_back(CLabeledWeightedRect(obj_bb, probability_map.at<float>(seed), label_it));
			}
			else
			{
				// if there is already a previous object of this class, take this id
				object_id = object_id_map.at<int>(ms_result);
				// and collect its weight
				objects_of_cur_label[object_id].weight_ += probability_map.at<float>(seed);
			}
			// set the cluster centers similarity raduis
			circle(object_id_map, ms_result, 5, Scalar(object_id), -1);
		}
		// append objects_of_cur_label to objects
		objects.insert(objects.end(), objects_of_cur_label.begin(), objects_of_cur_label.end());
	}

	//cout << endl << "Found " << objects.size() << " objects with the scores " << endl;
//	for(vector<CLabeledWeightedRect>::const_iterator it = objects.begin(); it != objects.end(); ++it)
//	{
//		cout << "Rect: " << it->rect_ << " with score: " << it->weight_ << " and label: " << it->label_ << endl;
//	}


#ifdef VERBOSE
	// visualize probability
	Mat prop_map_vis_fc1 = Mat::zeros(probability_map.size(), CV_32FC1);
	normalize(probability_map, prop_map_vis_fc1, 0, 1.0, NORM_MINMAX);

	Mat prop_map_vis_fc3;
	Mat t[] = {prop_map_vis_fc1, prop_map_vis_fc1, prop_map_vis_fc1};
	merge(t, 3, prop_map_vis_fc3);

	Mat vis_img = Mat::zeros(labels0.size(), CV_8UC3); // could be input image
	vis_img.convertTo(vis_img, CV_32FC3, 1.0/255);
	vis_img.setTo(Scalar(1.0,0,0), labels0 == 1);
	vis_img.setTo(Scalar(0,0,1.0), labels0 == 2);
	vis_img = vis_img.mul(prop_map_vis_fc3);
	Mat weighted_bg = Mat::zeros(labels0.size(), CV_8UC3); // could be input image;
	weighted_bg.convertTo(weighted_bg, CV_32FC3, 1.0/255);
	weighted_bg = weighted_bg.mul(Scalar(1.0,1.0,1.0)-prop_map_vis_fc3);
	vis_img = vis_img + weighted_bg;
	vis_img.convertTo(vis_img, CV_8UC3, 255.0);



	// Just a little visual Output
	for(vector<CLabeledWeightedRect>::const_iterator obj_it = objects.begin(); obj_it != objects.end(); ++obj_it)
	{
		//cout << "Canditate Rect: " << obj_it->rect_ << " with score: " << obj_it->weight_ << " and label: " << obj_it->label_ << endl;
		rectangle(vis_img, obj_it->rect_, Scalar(0,180,180), 2);
	}

	imshow("detect-result", vis_img);
	imwrite("detect_result.jpg", vis_img);

	//waitKey(0);
#endif // VERBOSE


	// analyze overlapping rectangles: sort by their weights and start with the highest rated

	// sort rectangles descending in their weights
	sort(objects.begin(), objects.end(), greaterLabeledWeightedRect);

	vector<CLabeledWeightedRect const*> non_overlap_rect_ptrs;

	// ignore overlapping results
	if(!objects.empty())
	{
		non_overlap_rect_ptrs.push_back(&objects.front());
		vector<CLabeledWeightedRect>::const_iterator obj_it;
		for(obj_it = objects.begin()+1; obj_it != objects.end(); ++obj_it)
		{
			bool do_overlap = false;
			vector<CLabeledWeightedRect const*>::const_iterator good_rect_it;
			for(good_rect_it = non_overlap_rect_ptrs.begin(); good_rect_it != non_overlap_rect_ptrs.end(); ++good_rect_it)
			{
				if(obj_it->getOverlapWith(**good_rect_it) > max_object_overlap)
					do_overlap = true;
			}

			if(!do_overlap)
			{
				non_overlap_rect_ptrs.push_back(&*obj_it);
			}
		}
	}

	// fill result vector array with vectors indexed by the labels
	results.clear();
	for(int label_cnt = 0; label_cnt < n_object_classes_; ++label_cnt)
		results.push_back(vector<Rect>());

	vector<CLabeledWeightedRect const*>::const_iterator good_rect_it;
	for(good_rect_it = non_overlap_rect_ptrs.begin(); good_rect_it != non_overlap_rect_ptrs.end(); ++good_rect_it)
	{
		results[(*good_rect_it)->label_-1].push_back((*good_rect_it)->rect_);
	}

}


void CPipelineController::train(const CommandParams& params)
{
	// initialize untrained pipe
	params_.str_classifier_   = params.str_classifier_;
	params_.vec_vec_channels_ = params.vec_vec_channels_;

	try{
		initializeFromParameters();
	}
	catch(PipeConfigExecption &e)
	{
		cout << e.what();
		exit(-1);
	}

	vector<string> filelist;
	if(!getFileList(params.str_imgset_, filelist))
	{
		exit(-1);
	}

	sort(filelist.begin(), filelist.end());
	vector<int> data_labels;

	int n_neg_samples = 0;
	int n_pos_samples = 0;

	// extract labels from filenames
	for(vector<string>::const_iterator f_it = filelist.begin(); f_it != filelist.end(); ++f_it)
	{
		stringstream ss_feature(*f_it);
		string str_label;
		getline(ss_feature, str_label, '_');
		int train_label;
		stringstream ss_label;
		ss_label << *(str_label.begin());
		ss_label >> train_label;
		data_labels.push_back(train_label);
		if(train_label == 0)
			++n_neg_samples;
		else
			++n_pos_samples;
	}

	//cout << "Expecting " << n_pos_samples + MAX_NUMBER_BG_SAMPLES << " samples" << endl;
	// if there are too many files: only take a subset of background patches (without duplicates)
	if(n_neg_samples > MAX_NUMBER_BG_SAMPLES)
	{
		cout << "Too many negative samples to fit in RAM. Rearranging to match " << MAX_NUMBER_BG_SAMPLES << " negative samples ..." << flush;
		srand(time(NULL));
		int n_removed_bg_samples = 0;
		int last_neg_idx_in_list = n_neg_samples-1;
		while(n_removed_bg_samples < n_neg_samples-MAX_NUMBER_BG_SAMPLES)
		{
			// random background sample index in list
			int rand_neg_idx = randRange<int>(0, last_neg_idx_in_list);

			// delete sample from list
			filelist.erase(filelist.begin()+rand_neg_idx);

			last_neg_idx_in_list--;
			n_removed_bg_samples++;
		}
		sort(filelist.begin(), filelist.end());

		// also remove labels
		data_labels.erase(data_labels.begin(), data_labels.begin() + n_removed_bg_samples);

		cout << " done!" << endl;

//		cout << "Got " << filelist.size() << " samples and " << data_labels.size() << " labels." << endl;
//
//		cout << "last data label: " << data_labels[MAX_NUMBER_BG_SAMPLES-1] << " next data label: " << data_labels[MAX_NUMBER_BG_SAMPLES] << endl;
//		cout << "last file name: " << filelist[MAX_NUMBER_BG_SAMPLES-1] << endl;
//		cout << "next file name: " << filelist[MAX_NUMBER_BG_SAMPLES] << endl;
		cout << flush;
	}

	cout << "Evaluating Pipeline... " << endl;

	CVisionModule* cmp = all_modules_[0]; // Current Module Ptr: grab a root module
	CVisionModule* lmp = NULL;            // Last Vision Module Pointer: used to identify correct buffer and converter inside a module
	vector<string>::const_iterator filename_it = filelist.begin();
	CVisionModule* current_path_root = cmp;
	//CVisionData* origin_sample = new CVisionData(imread(params.str_imgset_ + *filename_it), DATA_TYPE_IMAGE);
	CVisionData* current_data_ptr = createNewVisionDataObjectFromImageFile(params.str_imgset_ + *filename_it);

	bool finishedTraining = false;
	while(!finishedTraining)
	{
		cmp->bufferData(current_data_ptr, lmp);

		delete current_data_ptr; // cleanup old data should be OK, since it is buffered in the module and cv::Mat data is not deleted
		current_data_ptr = NULL;
		if(cmp->isTrained())
		{
			CVisionModule* missing_ancestor = cmp->getAncestorModuleFromWhichNoDataIsBuffered();
			if(!missing_ancestor)
			{
				current_data_ptr = cmp->exec(); // also clears Data Buffer of current Module
				lmp = cmp;
				cmp = cmp->getSuccessor();

				if(cmp == NULL) // reached the final module
					finishedTraining = true;

				continue;
			}
			else // if there is data missing: crawl the structure upwards the path to get data
			{
				while(missing_ancestor)
				{
					cmp = missing_ancestor;
					missing_ancestor = cmp->getAncestorModuleFromWhichNoDataIsBuffered();
					lmp = missing_ancestor; // is NULL when reached a root module
				}
				current_path_root = cmp;
				delete current_data_ptr;
				current_data_ptr = createNewVisionDataObjectFromImageFile(params.str_imgset_ + *filename_it);
				continue;
			}
		}
		else // reached a module that needs training: redo the same last path with each training sample
		{
			++filename_it;
			if(filename_it != filelist.end()) // if there are training samples left
			{
				for(int backspace_cnt = 0; backspace_cnt < 512; ++backspace_cnt)
					cout << '\b';
				cout << "processing file: " << params.str_imgset_ + *filename_it << flush;

				// load the next sample and go the same path
				delete current_data_ptr;
				current_data_ptr = createNewVisionDataObjectFromImageFile(params.str_imgset_ + *filename_it);
				cmp = current_path_root; // (will alternate if the path contains merging modules)
				lmp = NULL;
				continue;
			}
			else // reached the module to be trained with the last sample
			{
				cout << endl;
				// see if there are another paths towards this module
				CVisionModule* missing_ancestor = cmp->getAncestorModuleFromWhichNoDataIsBuffered();
				if(!missing_ancestor)
				{
					// if not, train the module - all data should be in the buffer
					cmp->setDataLabels(CVisionData(Mat(data_labels), DATA_TYPE_VECTOR));
					cmp->train(); // also clears Data Buffer of current Module
					cmp->setAsTrained();
					cmp = current_path_root; // restart from root module
					lmp = NULL;
				}
				else // if there is another path: crawl the structure upwards this path to get data using the same sample
				{
					while(missing_ancestor)
					{
						cmp = missing_ancestor;
						missing_ancestor = cmp->getAncestorModuleFromWhichNoDataIsBuffered();
						lmp = missing_ancestor; // is NULL when reached a root module
					}
					// now go the other path
					current_path_root = cmp;
				}
				// load the first sample
				filename_it = filelist.begin();
				delete current_data_ptr;
				current_data_ptr = createNewVisionDataObjectFromImageFile(params.str_imgset_ + *filename_it);
				continue;
			}
		}
	}

	// get number of different classes; data_labels is not needed anymore
	vector<int>::iterator t_it;
	t_it = unique(data_labels.begin(), data_labels.end());
	data_labels.resize(distance(data_labels.begin(), t_it));
	sort(data_labels.begin(), data_labels.end());
	t_it = unique(data_labels.begin(), data_labels.end());
	n_object_classes_ = distance(data_labels.begin(), t_it) - 1; // subtract background class

	cout << "Number of different object classes: " << n_object_classes_ << endl;

}

CVisionData* CPipelineController::createNewVisionDataObjectFromImageFile(const string& filename)
{
	Mat image = imread(filename);
	resize(image, image, Size(SWIN_SIZE, SWIN_SIZE));
	return new CVisionData(image, DATA_TYPE_IMAGE);
}


void CPipelineController::printConfig()
{
	cout << endl;
	cout << "Vision Pipeline Configuration:" << endl;
	cout << "------------------------------" << endl;

//	CVisionModule* cur_mod = all_modules_[0]; // start with a root module
//	map<string,CVisionModule*> printed_modules;
//	while(printed_modules.size() != all_modules_.size())
//	{
//		// if module was not already printed, print it
//		if(printed_modules.find(cur_mod->getModuleID()) == printed_modules.end())
//		{
//			printed_modules[cur_mod->getModuleID()] = cur_mod;
//			cout << "(" << cur_mod->getPrintName() << ")";
//		}
//
//		// go to the successor module
//		cur_mod = cur_mod->getSuccessor();
//
//		// reached the end module
//		if(cur_mod == NULL)
//		{
//			unsigned int root_id_cnt = 0;
//			cur_mod = all_modules_[0];
//			// look for next root module that was not printed
//			while(++root_id_cnt < all_modules_.size())
//			{
//				cur_mod = all_modules_[root_id_cnt];
//				if(printed_modules.find(cur_mod->getModuleID()) == printed_modules.end() && cur_mod->isRoot())
//				{
//					break;
//				}
//			}
//			cout << endl;
//		}
//		else
//		{
//			cout << setw(10);
//		}
//	}
//	cout << endl;


	for(vector<vector<string> >::iterator ch_it = params_.vec_vec_channels_.begin(); ch_it != params_.vec_vec_channels_.end(); ++ch_it)
	{
		cout << "Channel" << distance(params_.vec_vec_channels_.begin(), ch_it) + 1 << ":" << endl;
		for(vector<string>::iterator mod_it = ch_it->begin(); mod_it != ch_it->end(); ++mod_it)
		{
			cout << " -" << *mod_it << endl;
		}
	}

//	for (vector<CVisionModule*>::iterator mod_it = all_modules_.begin(); mod_it != all_modules_.end(); ++mod_it)
//	{
//		cout << "-Feature Channel " << distance(all_modules_.begin(), mod_it) << ": " << endl;
//		cout << "   -" << (*mod_it)->getPrintName() << endl;
//	}

	cout << "Final Classifier: " << params_.str_classifier_ << endl;
	cout << endl;

}

void CPipelineController::load(const string& filename)
{
	// create new config file
	FileStorage fs;

	fs.open(filename, FileStorage::READ);
	if(!fs.isOpened())
	{
		cerr << "Config file not found! Exiting." << endl;
		exit(-1);
	}

	int swin_size;
	fs[CONFIG_NAME_SWIN_SIZE] >> swin_size;
	if(swin_size != SWIN_SIZE)
	{
		cerr << "ERROR: Loaded config file was trained with different Sliding Window size!" << endl;
		exit(-1);
	}


	// get all channel ids
	// TODO: change to FileNode Iterator!!!
	for(int ch_nr = 1; ch_nr <= 6; ++ch_nr)
	{
		stringstream ch_nr_ss;
		ch_nr_ss << "-" << ch_nr;
		vector<string> channel_ids;
		fs[CONFIG_NAME_CHANNEL + ch_nr_ss.str()] >> channel_ids;
		if(channel_ids.empty())
			break;

		params_.vec_vec_channels_.push_back(channel_ids);
	}

	fs[CONFIG_NAME_CLASSIFIER] >> params_.str_classifier_;

	try{
		initializeFromParameters();
	}
	catch(PipeConfigExecption &e)
	{
		cout << e.what();
		exit(-1);
	}

	fs[CONFIG_NAME_NUM_CLASSES] >> n_object_classes_;

	for (vector<CVisionModule*>::const_iterator mod_it = all_modules_.begin(); mod_it != all_modules_.end(); ++mod_it)
	{
		(*mod_it)->load(fs);
		(*mod_it)->setAsTrained();
	}

	fs.release();

	//cout << "Pipeline loaded." << endl;
}


void CPipelineController::save(const string& filename)
{
	FileStorage fs;
	// open config file
	fs.open(filename, FileStorage::WRITE);
	if(!fs.isOpened())
	{
		cerr << "Failed to access file \"" << filename << "\". Check write permissions." << endl;
		exit(-1);
	}

	fs << CONFIG_NAME_SWIN_SIZE << SWIN_SIZE;

	// Save Pipeline configuration
	for(vector<vector<string> >::iterator ch_it = params_.vec_vec_channels_.begin(); ch_it != params_.vec_vec_channels_.end(); ++ch_it)
	{
		stringstream ch_nr;
		ch_nr << "-" << distance(params_.vec_vec_channels_.begin(), ch_it)+1;
		fs << CONFIG_NAME_CHANNEL + ch_nr.str() << *ch_it;
	}

	fs << CONFIG_NAME_CLASSIFIER << params_.str_classifier_;
	fs << CONFIG_NAME_NUM_CLASSES << n_object_classes_;

	for (vector<CVisionModule*>::const_iterator mod_it = all_modules_.begin(); mod_it != all_modules_.end(); ++mod_it)
	{
		(*mod_it)->save(fs);
	}


	fs.release();

	//cout << "Pipeline saved." << endl;
}

