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

	bitset<16> x1(CV_8UC1), x2(CV_8UC3), x3(CV_32FC1), x4(CV_32FC2), x5(CV_32SC1), x6(CV_32SC2);
	cout << "CV_8UC1: " << CV_8UC1 << "    is: " << x1 << endl;
	cout << "CV_8UC3: " << CV_8UC3 << "   is: " << x2 << endl;
	cout << "CV_32FC1: " << CV_32FC1 << "   is: " << x3 << endl;
	cout << "CV_32FC2: " << CV_32FC3 << "  is: " << x4 << endl;
	cout << "CV_32SC1: " << CV_32SC1 << "   is: " << x5 << endl;
	cout << "CV_32SC2: " << CV_32SC3 << "  is: " << x6 << endl;

	// Set up Feature Channels for the simple structure
	bool is_root = true;
	vector<CVisionModule*> end_modules_of_channels;
	vector<vector<string> >::const_iterator ch_str_it;
	for(ch_str_it = params_.vec_vec_channels_.begin(); ch_str_it != params_.vec_vec_channels_.end(); ++ch_str_it)
	{
		is_root = true;
		CVisionModule* ancestor_mod = NULL;
		vector<string>::const_iterator mod_id_it;
		for(mod_id_it = ch_str_it->begin(); mod_id_it != ch_str_it->end(); ++mod_id_it)
		{
			CVisionModule* cur_mod = NULL;
			// Pre-processing Methods
			if(*mod_id_it == ID_CANNY)
				cur_mod = new CBinaryContours(is_root);
			if(*mod_id_it == ID_GRADIENT)
				cur_mod = new CGradientImage(is_root);
			if(*mod_id_it == ID_DISTTR)
				cur_mod = new CDistanceTransform(is_root);

			// Feature Extractor Methods
			if(*mod_id_it == ID_HOG)
				cur_mod = new CHog(is_root);
			if(*mod_id_it == ID_BRIEF)
				cur_mod = new COwnBrief(is_root);

			// Subspace methods
			if(*mod_id_it == ID_PCA)
				cur_mod = new CPCA(is_root);

			// Classifiers: TODO: re-implementation of classifiers needed (not just class label as output)
			if(*mod_id_it == ID_SVM)
				cur_mod = new CSVM(is_root);
			if(*mod_id_it == ID_RF)
				cur_mod = new CRandomForest(is_root);

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

	if(params_.str_classifier_ == ID_SVM)
		final_classifier = new CSVM(is_root);
	if(params_.str_classifier_ == ID_RF)
		final_classifier = new CRandomForest(is_root);

	if(!final_classifier)
		throw(PipeConfigExecption()); // unknown ID given!!!

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

//#pragma omp parallel for
		for(win_roi.x = 0; win_roi.x < input_img.cols-SWIN_SIZE; win_roi.x += 1)
		{
			//CVisionData window(input_img(win_roi), DATA_TYPE_IMAGE);
			//			window->show();
			//			waitKey(3);

			// Execute each Pipeline and concatenate results
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
			if(result_class)
			{
				Point2i win_center(win_roi.x + win_roi.width/2, win_roi.y + win_roi.height/2);
				mat_detect_labels.at<uchar>(win_center) = result_class;
				mat_detect_prop.at<float>(win_center) = current_data_ptr->data().at<float>(0,1);
			}

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

#ifdef VERBOSE
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
	imwrite("detect-result.png", vis_img);
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
	initializeFromParameters();

	vector<string> filelist;
	if(getFileList(params.str_imgset_, filelist))
	{
		sort(filelist.begin(), filelist.end());

		CVisionModule* cmp = all_modules_[0]; // Current Module Ptr: grab a root module
		CVisionModule* lmp = NULL;            // Last Vision Module Pointer: used to identify correct buffer and converter inside a module
		vector<string>::const_iterator filename_it = filelist.begin();
		CVisionModule* current_path_root = cmp;
		//CVisionData* origin_sample = new CVisionData(imread(params.str_imgset_ + *filename_it), DATA_TYPE_IMAGE);
		CVisionData* current_data_ptr = createNewVisionDataObjectFromImageFile(params.str_imgset_ + *filename_it);
		vector<int> data_labels;

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
				// first: collect the class label during the first iteration through the training data
				if(data_labels.size() != filelist.size())
				{
					stringstream ss_feature(*filename_it);
					string str_label;
					getline(ss_feature, str_label, '_');
					int train_label;
					stringstream ss_label;
					ss_label << *(str_label.begin());
					ss_label >> train_label;
					data_labels.push_back(train_label);
				}

				++filename_it;
				if(filename_it != filelist.end()) // the last there are training samples left
				{
					// load the next sample and go the same path
					delete current_data_ptr;
					current_data_ptr = createNewVisionDataObjectFromImageFile(params.str_imgset_ + *filename_it);
					cmp = current_path_root; // (will alternate if the path contains merging modules)
					lmp = NULL;
					continue;
				}
				else // reached the module to be trained with the last sample
				{
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

//
//		// get number of different classes; train_labels is not needed anymore
//		vector<int>::iterator t_it;
//		t_it = unique(train_labels.vec_.begin(), train_labels.vec_.end());
//		train_labels.vec_.resize(distance(train_labels.vec_.begin(), t_it));
//		sort(train_labels.vec_.begin(), train_labels.vec_.end());
//		t_it = unique(train_labels.vec_.begin(), train_labels.vec_.end());
//		n_object_classes_ = distance(train_labels.vec_.begin(), t_it) - 1; // subtract background class
//
//		cout << "Number of different object classes: " << n_object_classes_ << endl;
	}
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


	// get all channel ids
	for(int ch_nr = 1; ch_nr <= 5; ++ch_nr)
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

	initializeFromParameters();

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

