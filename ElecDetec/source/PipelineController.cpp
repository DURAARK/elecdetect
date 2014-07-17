/*
 * PipelineController.cpp
 *
 *  Created on: Jul 3, 2014
 *      Author: test
 */

#include "PipelineController.h"


CPipelineController::CPipelineController() : feature_length_(0), n_object_classes_(0)
{
}


CPipelineController::~CPipelineController()
{
	deletePipe();
}


void CPipelineController::deletePipe()
{
	// clean up modules
	for (vector<CVisionModule*>::const_iterator mod_it = v_modules_.begin(); mod_it != v_modules_.end(); ++mod_it)
	{
		delete *mod_it;
	}
	v_modules_.clear();
}


void CPipelineController::initializeFromParameters() throw (PipeConfigExecption)
{
	deletePipe(); // delete Pipe if already set up

	// Set up Preprocessing
	vector<string>::const_iterator mod_id_it;
	for(mod_id_it = params_.vec_preproc_.begin(); mod_id_it != params_.vec_preproc_.end(); ++mod_id_it)
	{
		if(*mod_id_it == ID_CANNY)
			v_modules_.push_back(new CBinaryContours());
		if(*mod_id_it == ID_GRADIENT)
			v_modules_.push_back(new CGradientImage());
		if(*mod_id_it == ID_DISTTR)
			v_modules_.push_back(new CDistanceTransform());
	}

	// for now: just use one feature extractor                                      !!!!!!!!!
	for(mod_id_it = params_.vec_feature_.begin(); mod_id_it != params_.vec_feature_.begin()+1; ++mod_id_it)
	{
		CFeatureExtractorModule* f_ptr = NULL;
		if(*mod_id_it == ID_HOG)
			f_ptr = new CHog();
		if(*mod_id_it == ID_BRIEF)
			f_ptr = new COwnBrief();

		if(!f_ptr)
		{
			// no feature extractor -> use whole image as feature
			cout << "no feature method specified. using whole image as feature vector" << endl;
			f_ptr = new CDummyFeature();
		}

		feature_length_ += f_ptr->getFeatureLength();
		v_modules_.push_back(f_ptr);
	}

	// TODO later: setup multiple feature-vectors


	// Add classifier to pipeline
	CClassifierModule* c_ptr = NULL;
	if(params_.str_classifier_ == ID_SVM)
		c_ptr = new CSVM();
	if(params_.str_classifier_ == ID_RF)
		c_ptr = new CRandomForest();

	if(!c_ptr)
		throw(PipeConfigExecption());

	v_modules_.push_back(c_ptr);

	printConfig();


}


void CPipelineController::test(const Mat& input_img, vector<vector<Rect> >& bb_results)
{
	Mat mat_detect_labels = Mat::zeros(input_img.size(), CV_8UC1); // discrete labels
	Mat mat_detect_prop = Mat::zeros(input_img.size(), CV_32FC1);   // probability map of the detection results
	cout << "Analyzing image...        " << flush;
	float progress = 0.0;
	// Sliding window
	Rect win_roi(0, 0, SWIN_SIZE, SWIN_SIZE);
	for(win_roi.y = 0; win_roi.y < input_img.rows-SWIN_SIZE; win_roi.y += 1)
	{
		progress = 100.0f*win_roi.y/(input_img.rows-SWIN_SIZE);
		for(int backspace_cnt = 0; backspace_cnt < 6; ++backspace_cnt)
			cout << '\b';
		cout << setfill('0');
		cout << setw(5) << setiosflags(ios::fixed) << setprecision(2) << progress << "%" << flush;

		for(win_roi.x = 0; win_roi.x < input_img.cols-SWIN_SIZE; win_roi.x += 1)
		{
			CVisionData* window = new CMat();
			((CMat*)window)->mat_ = input_img(win_roi);
//			window->show();
//			waitKey(3);
			std::vector<CVisionData*> v_data;
			v_data.push_back(window);

			// Execute Pipeline
			try
			{
				for (std::vector<CVisionModule*>::const_iterator mod_it = v_modules_.begin(); mod_it != v_modules_.end(); ++mod_it)
				{
					(*mod_it)->exec(v_data);
				}
			}
			catch (VisionDataTypeException& e)
			{
				cout << e.what();
			}

			CWeightedScalar<int>* result = reinterpret_cast<CWeightedScalar<int>* >(v_data.back());

			if(result->val_)
			{
				Point2i win_center(win_roi.x + win_roi.width/2, win_roi.y + win_roi.height/2);
				mat_detect_labels.at<uchar>(win_center) = result->val_;
				mat_detect_prop.at<float>(win_center) = result->weight_;
			}

			// Cleanup data
			for (std::vector<CVisionData*>::const_iterator data_it = v_data.begin(); data_it != v_data.end(); ++data_it)
			{
				delete *data_it;
			}
			v_data.clear();
		}
	}


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
	const float max_object_overlap = 0.1;

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
					getStructuringElement(MORPH_ELLIPSE, Size(OPENING_SIZE, OPENING_SIZE) ) );

			labels0.copyTo(labels, closed_mask); // aware that higher labels may replace lower labels

			// and Gauß-filter to estimate probability
			Mat closed_mask_float = Mat::zeros(labels0.size(), CV_32FC1);
			closed_mask.convertTo(closed_mask_float, CV_32FC1, 1.0/255);
			int ksize = 2*static_cast<int>(SWIN_SIZE/2)+1;
			GaussianBlur(closed_mask_float, closed_mask_float, Size(ksize, ksize), (float)SWIN_SIZE/15.0);

			closed_mask_float.copyTo(probability_map, closed_mask); // aware that higher labels may replace lower labels
		}

		Mat vis_img;

		vis_img = Mat::zeros(labels0.size(), CV_8UC3);
		vis_img.setTo(Scalar(255,0,0), labels0 == 1);
		vis_img.setTo(Scalar(0,0,255), labels0 == 2);

		imshow("Original Labels", vis_img);

		vis_img = Mat::zeros(labels0.size(), CV_8UC3);
		vis_img.setTo(Scalar(255,0,0), labels == 1);
		vis_img.setTo(Scalar(0,0,255), labels == 2);

		imshow("Closed Labels", vis_img);

		// - visualization ------------------------
		Mat prop_map_vis = Mat::zeros(probability_map.size(), CV_32FC1);
		normalize(probability_map, prop_map_vis, 0, 1.0, NORM_MINMAX);
		vis_img = Mat::zeros(labels0.size(), CV_8UC3);
		prop_map_vis.convertTo(vis_img, CV_8UC3, 255.0);
		imshow("Probabilities of Labels", vis_img);
		// ----------------------------------------

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
	imwrite("detect-result.png", vis_img);



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
	params_.str_classifier_ = params.str_classifier_;
	params_.vec_feature_ = params.vec_feature_;
	params_.vec_preproc_ = params.vec_preproc_;
	initializeFromParameters();

	vector<string> filelist;
	if(getFileList(params.str_imgset_, filelist))
	{
		// Allocate Space for Training Data
		CMat train_data;
		train_data.mat_ = Mat(filelist.size(), feature_length_, CV_32FC1);
		//cout << "Feature Size: " << train_data.mat_.rows << " x " << train_data.mat_.cols << endl << flush;

		CVector<int> train_labels;
		int sample_cnt = 0;
		sort(filelist.begin(), filelist.end());
		vector<string>::const_iterator file_it;
		for(file_it = filelist.begin(); file_it != filelist.end(); ++file_it)
		{
			CMat* training_sample_ptr = new CMat();
			training_sample_ptr->mat_ = cv::imread(params.str_imgset_ + *file_it);
			//training_sample_ptr->show();
			std::vector<CVisionData*> v_data;
			v_data.push_back(training_sample_ptr);

			// Execute Pipeline exclusive classifier
			for (std::vector<CVisionModule*>::const_iterator mod_it = v_modules_.begin(); (*mod_it)->getType() != MOD_TYPE_CLASSIFIER; ++mod_it)
			{
				(*mod_it)->exec(v_data);
//				v_data.back()->show();
//				waitKey(0);
//				if((*mod_it)->getType() == MOD_TYPE_PREPROC)
//				{
//					imwrite("out.png", reinterpret_cast<CMat*>(v_data.back())->mat_*4);
//				}
			}
			// feature is now at the end of v_data and was created with new
			// create temporary matrix header with the feature vector data (without copying)
			Mat temp(((CVector<float>*)v_data.back())->vec_, false);
			// reshape to row-vector and add feature vector to all samples (with copying data)
			temp.reshape(0,1).copyTo(train_data.mat_.row(sample_cnt));

			// clean up data of current sample
			for (std::vector<CVisionData*>::const_iterator data_it = v_data.begin(); data_it != v_data.end(); ++data_it)
			{
				delete *data_it;
			}
			v_data.clear();

			// collect class label
			stringstream ss_feature(*file_it);
			string str_label;
			getline(ss_feature, str_label, '_');
			int train_label;
			stringstream ss_label;
			ss_label << *(str_label.begin());
			ss_label >> train_label;
			train_labels.vec_.push_back(train_label);
			//cout << *file_it << " : " << train_label << endl << flush; // TODO: check if labels are extarcted correctly
			// waitKey(0);

			sample_cnt++;
		}

		//cout << train_data.mat_ << endl << "<- Train Data" << endl;
		// Select classifier TODO: test for classifier type
		CClassifierModule* classifier = reinterpret_cast<CClassifierModule*>(v_modules_.back());

		// Perform Training
		classifier->train(train_data, train_labels);

		// get number of different classes
		vector<int>::iterator t_it;
		t_it = unique(train_labels.vec_.begin(), train_labels.vec_.end());
		train_labels.vec_.resize(distance(train_labels.vec_.begin(), t_it));
		sort(train_labels.vec_.begin(), train_labels.vec_.end());
		t_it = unique(train_labels.vec_.begin(), train_labels.vec_.end());
		n_object_classes_ = distance(train_labels.vec_.begin(), t_it) - 1; // subtract background class

//		for(vector<int>::iterator temp_it = train_labels.vec_.begin(); temp_it != train_labels.vec_.end(); ++temp_it)
//			cout << "unique label: " << *temp_it << endl;

		cout << "Number of different object classes: " << n_object_classes_ << endl;

	}
}


void CPipelineController::printConfig()
{
	cout << endl;
	cout << "Vision Pipeline Configuration:" << endl;
	cout << "------------------------------" << endl;

	int mod_type = 0;
	for (vector<CVisionModule*>::const_iterator mod_it = v_modules_.begin(); mod_it != v_modules_.end(); ++mod_it)
	{
		if(mod_type != (*mod_it)->getType())
		{
			if((*mod_it)->getType() == MOD_TYPE_PREPROC)
				cout << "Preprocessing:" << endl;
			if((*mod_it)->getType() == MOD_TYPE_FEATURE)
				cout << "Feature Descriptor:" << endl;
			if((*mod_it)->getType() == MOD_TYPE_CLASSIFIER)
				cout << "Classifier:" << endl;

			mod_type = (*mod_it)->getType();
		}
		cout << " -" << (*mod_it)->getName() << endl;
	}
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

	fs[CONFIG_NAME_PREPROC] >> params_.vec_preproc_;
	fs[CONFIG_NAME_FEATURE] >> params_.vec_feature_;
	fs[CONFIG_NAME_CLASSIFIER] >> params_.str_classifier_;

	initializeFromParameters();

	fs[CONFIG_NAME_NUM_CLASSES] >> n_object_classes_;

	for (vector<CVisionModule*>::iterator mod_it = v_modules_.begin(); mod_it != v_modules_.end(); ++mod_it)
	{
		(*mod_it)->load(fs);
	}
	fs.release();

	cout << "Pipeline loaded." << endl;
}


void CPipelineController::save(const string& filename)
{
	// generate classifier config filename
//	std::size_t xmlpos = params_.str_configfile_.find(".xml");
//	string str_classifier_configfile = params_.str_configfile_.substr(0, xmlpos);
//	str_classifier_configfile += FILENAME_CLASSIFIER_POSTFIX;
//	str_classifier_configfile += ".xml";

	FileStorage fs;
	// open config file
	fs.open(filename, FileStorage::WRITE);
	if(!fs.isOpened())
	{
		cerr << "Failed to access file \"" << filename << "\". Check write permissions." << endl;
		exit(-1);
	}

	// Save Pipeline configuration

	fs << CONFIG_NAME_PREPROC << params_.vec_preproc_;
	fs << CONFIG_NAME_FEATURE << params_.vec_feature_;
	fs << CONFIG_NAME_CLASSIFIER << params_.str_classifier_;

	fs << CONFIG_NAME_NUM_CLASSES << n_object_classes_;

	for (vector<CVisionModule*>::const_iterator mod_it = v_modules_.begin(); mod_it != v_modules_.end(); ++mod_it)
	{
		(*mod_it)->save(fs);
	}

	fs.release();

	cout << "Pipeline saved." << endl;
}

