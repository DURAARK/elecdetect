/*
 * ElecDetec: AlgorithmController.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#include "AlgorithmController.h"


CAlgorithmController::CAlgorithmController() :
    filter_horiz_(NULL),
    filter_vert_(NULL),
    hog_extractor_(NULL),
    ori_haar_extractor_(NULL),
    haar_extractor_(NULL),
    gof_extractor_(NULL),
    random_forest_(NULL),
    bootstrap_nstages_(_MAX_BOOTSTRAP_STAGES_),
    bootstrap_train_error_threshold_(BOOTSTRAP_MIN_ERROR)
{
    reInitAlgorithm();
}


CAlgorithmController::~CAlgorithmController()
{
    clearAlgorithm();
}

void CAlgorithmController::clearAlgorithm()
{
    if(filter_horiz_) delete filter_horiz_;
    if(filter_vert_) delete filter_vert_;
    if(hog_extractor_) delete hog_extractor_;
    if(ori_haar_extractor_) delete ori_haar_extractor_;
    if(haar_extractor_) delete haar_extractor_;
    if(gof_extractor_) delete gof_extractor_;
    if(random_forest_) delete random_forest_;
    class_labels_.clear();
}

void CAlgorithmController::reInitAlgorithm()
{
    clearAlgorithm();

    filter_horiz_ = new COrientationFilter(COrientationFilter::HORIZ);
    filter_vert_ = new COrientationFilter(COrientationFilter::VERT);
    hog_extractor_ = new CHog();
    ori_haar_extractor_ = new CHaarWavelets();
    haar_extractor_ = new CHaarWavelets();
    gof_extractor_ = new CGradientOrientationFeatures();
    random_forest_ = new CRandomForest();
}


void CAlgorithmController::detect(const Mat& test_img, const string& test_img_name, vector<Mat>& result_probs, vector<CLASS_LABEL_TYPE>& labels)
throw(FileAccessExecption)
{
    result_probs.clear();
    if(test_img.empty())
    {
        string msg = "Failed to read image file \"" + test_img_name + "\"!";
        throw(FileAccessExecption(msg.c_str()));
    }

    Mat mat_to_analyze = Mat::zeros(test_img.size(), CV_8UC1);           // Binary image that codes which pixels (BB center points)
                                                                          // are going to be analyzed
    const Point2i center_shift(_PATCH_WINDOW_SIZE_/2, _PATCH_WINDOW_SIZE_/2);
    const float refine_search_threshold = (1.0f/class_labels_.size())/2.0f;
    const string analyze_msg = "Analyzing image... ";

    // probability images for each class (also background class)
    result_probs.clear();
    for(uint i = 0; i < class_labels_.size(); ++i)
    {
        result_probs.push_back(Mat::zeros(test_img.size(), CV_32FC1));
    }
    labels = class_labels_;

	// Scan Grid 'sg'
	vector<int> sg_x, sg_y;
    linspace<int>(sg_x, 0, test_img.cols-_PATCH_WINDOW_SIZE_-1, ceil(((float)test_img.cols-_PATCH_WINDOW_SIZE_)/(float)OPENING_SIZE)+2);
    linspace<int>(sg_y, 0, test_img.rows-_PATCH_WINDOW_SIZE_-1, ceil(((float)test_img.rows-_PATCH_WINDOW_SIZE_)/(float)OPENING_SIZE)+2);
//	cout << "image is " << input_img.cols << " x " << input_img.rows << endl;
//	cout << "linspace x n_elements: " << sg_x.size() << " and y: " << sg_y.size() << endl;

    vector<Point> tl_scan_points;
    tl_scan_points.reserve(mat_to_analyze.rows*mat_to_analyze.cols);
	for(vector<int>::const_iterator y_it = sg_y.begin(); y_it != sg_y.end(); ++y_it)
	{
		for(vector<int>::const_iterator x_it = sg_x.begin(); x_it != sg_x.end(); ++x_it)
		{
            tl_scan_points.push_back(Point(*x_it, *y_it));
            mat_to_analyze.at<uchar>(Point(*x_it, *y_it) + center_shift) = 255;
		}
	}

    int progress_refresh_cnt = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int scan_pt_cnt = 0; scan_pt_cnt < tl_scan_points.size(); ++scan_pt_cnt)
    {

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if(progress_refresh_cnt++ >= 100)
            {
                const float progress = 100.0f*(float)scan_pt_cnt/tl_scan_points.size();

                for(int backspace_cnt = 0; backspace_cnt < 1024; ++backspace_cnt)
                    cout << '\b';

                cout << analyze_msg << test_img_name << ": " << setfill('0') << setw(5) << setiosflags(ios::fixed) << setprecision(2) << progress << "%" << flush;
                progress_refresh_cnt = 0;
            }
        }

        // a query point array per thread: necessary for overall dynamically growing query points
        vector<Point> query_points;
        query_points.push_back(tl_scan_points[scan_pt_cnt]);

        for(uint qpt_cnt = 0; qpt_cnt < query_points.size(); ++qpt_cnt)
        {
            Rect win_roi(0, 0, _PATCH_WINDOW_SIZE_, _PATCH_WINDOW_SIZE_);
            win_roi.x = query_points[qpt_cnt].x;
            win_roi.y = query_points[qpt_cnt].y;

            // Predict the class probability of the window patch
            vector<float> feature_vec;
            extractFeatures(test_img(win_roi), feature_vec);

            vector<CRandomForest::WeightedLabel> pred_results;
            random_forest_->predict(feature_vec, pred_results);

            float highest_object_prob = 0.0f;
            const Point2i win_center = win_roi.tl() + center_shift;//win_roi.x + win_roi.width/2, win_roi.y + win_roi.height/2);
            // for each class in the result (rows in the result data)
            for(vector<CRandomForest::WeightedLabel>::const_iterator class_pred_it = pred_results.begin();
                class_pred_it != pred_results.end(); ++class_pred_it)
            {
                // get result class idx and probability
                const CLASS_LABEL_TYPE class_label = class_pred_it->label_;
                const float prob = class_pred_it->weight_;

                // store the weight in the corresponding image at the right position
                const int class_label_idx = classLabel2Idx(class_label);
                result_probs[class_label_idx].at<float>(win_center) = prob;

                // get highest object prob
                if(class_label != _BACKGROUND_LABEL_)
                    if(prob > highest_object_prob)
                        highest_object_prob = prob;
            }

            // if an object is probable at this position
            if(highest_object_prob > refine_search_threshold)
            {
                // add surrounding (TOP LEFT) points to analyze
                Point from(win_roi.x - OPENING_SIZE, win_roi.y - OPENING_SIZE);
                from.x = from.x >= 0 ? from.x : 0;
                from.y = from.y >= 0 ? from.y : 0;

                Point to(win_roi.x + OPENING_SIZE, win_roi.y + OPENING_SIZE);
                to.x = to.x <= test_img.cols - _PATCH_WINDOW_SIZE_ - 1 ? to.x : test_img.cols - _PATCH_WINDOW_SIZE_ - 1;
                to.y = to.y <= test_img.rows - _PATCH_WINDOW_SIZE_ - 1 ? to.y : test_img.rows - _PATCH_WINDOW_SIZE_ - 1;

                for(int x = from.x; x <= to.x; ++x)
                {
                    for(int y = from.y; y <= to.y; ++y)
                    {
                        Point cur_pt(x,y);

#ifdef _OPENMP
#pragma omp critical
#endif
                        {
                            if(!mat_to_analyze.at<uchar>(cur_pt+center_shift)) // add the points just once
                            {
                                query_points.push_back(cur_pt);
                                mat_to_analyze.at<uchar>(cur_pt+center_shift) = 255;
                            }
                        }
                    }
                }
            }
        }
    }

    // fillup undiscovered points as background
    for(uint i = 0; i < class_labels_.size(); ++i)
    {
        if(class_labels_[i] == _BACKGROUND_LABEL_)
        {
            result_probs[i].setTo(1.0f, 255-mat_to_analyze);
            //PAUSE_AND_SHOW(result_probs[i]);
            break;
        }
    }

    for(int backspace_cnt = 0; backspace_cnt < 1024; ++backspace_cnt)
		cout << '\b';
    cout << analyze_msg << test_img_name << " finished." << endl;
}


void CAlgorithmController::train(const vector<string>& trainfiles) throw(FileAccessExecption)
{
	srand(time(NULL));

    vector<string> filelist = trainfiles; // make a writable copy of filename vector

	sort(filelist.begin(), filelist.end());
    vector<CLASS_LABEL_TYPE> data_labels;

    int n_neg_samples = 0;
    int n_pos_samples = 0;

    // extract labels from filenames
    for(vector<string>::const_iterator f_it = filelist.begin(); f_it != filelist.end(); ++f_it)
    {
        // extract filename from path
        vector<string> path_parts;
        path_parts = splitStringByDelimiter(*f_it, FOLDER_CHAR);
        string filename = path_parts.back();

        // extract label from filename
        vector<string> name_parts;
        name_parts = splitStringByDelimiter(filename, _LABEL_DELIMITER_);
        string str_label = name_parts.front();

        // convert label string to CLASS_LABEL_TYPE datatype
        CLASS_LABEL_TYPE train_label;
        stringstream ss_label;
        ss_label << *(str_label.begin());
        ss_label >> train_label;
        data_labels.push_back(train_label);
        if(train_label == _BACKGROUND_LABEL_)
            ++n_neg_samples;
        else
            ++n_pos_samples;
    }

//    //cout << "Expecting " << n_pos_samples + MAX_NUMBER_BG_SAMPLES << " samples" << endl;
//    // if there are too many files: only take a subset of background patches (without duplicates)
//    if(n_neg_samples > MAX_NUMBER_BG_SAMPLES)
//    {
//        cout << "Too many negative samples to fit in RAM. Rearranging to match " << MAX_NUMBER_BG_SAMPLES << " negative samples ..." << flush;
//        int n_removed_bg_samples = 0;
//        int last_neg_idx_in_list = n_neg_samples-1;
//        while(n_removed_bg_samples < n_neg_samples-MAX_NUMBER_BG_SAMPLES)
//        {
//            // random background sample index in list
//            int rand_neg_idx = randRange<int>(0, last_neg_idx_in_list);

//            // delete sample from list
//            filelist.erase(filelist.begin()+rand_neg_idx);

//            last_neg_idx_in_list--;
//            n_removed_bg_samples++;
//        }
//        sort(filelist.begin(), filelist.end());

//        // also remove labels
//        data_labels.erase(data_labels.begin(), data_labels.begin() + n_removed_bg_samples);

//        cout << " done!" << endl;

////		cout << "Got " << filelist.size() << " samples and " << data_labels.size() << " labels." << endl;
////
////		cout << "last data label: " << data_labels[MAX_NUMBER_BG_SAMPLES-1] << " next data label: " << data_labels[MAX_NUMBER_BG_SAMPLES] << endl;
////		cout << "last file name: " << filelist[MAX_NUMBER_BG_SAMPLES-1] << endl;
////		cout << "next file name: " << filelist[MAX_NUMBER_BG_SAMPLES] << endl;
//        cout << flush;
//    }


    // Bootstrapping
    uint nsamples = filelist.size();
    vector<uint> hard_sample_idxes;
    vector<uint> selected_samples_indxes;
    float train_error = 1.0;
    int stages_cnt = 0;
    // do boostraping until train_error reaches a threshold or the number of stages is reached
    // or the maximum amount of hard examples is reached
    while(train_error > bootstrap_train_error_threshold_ &&
          stages_cnt++ < bootstrap_nstages_ &&
          hard_sample_idxes.size() + selected_samples_indxes.size() < nsamples)
    {
        // initialize algorithm parts completely new to clear already learned data
        this->reInitAlgorithm();

        cout << "Starting bootstrapping stage " << stages_cnt << "..." << endl;

        // select new training subset and add all hard examples
        bootstrap_getRandomlySelectedSampleIndices(data_labels, selected_samples_indxes);
        selected_samples_indxes.insert(selected_samples_indxes.end(), hard_sample_idxes.begin(), hard_sample_idxes.end());

        int processing_file_msg_cnt = 0;

        vector<vector<float> > training_data;
        vector<CLASS_LABEL_TYPE> training_labels;
        training_data.reserve(selected_samples_indxes.size());
        training_labels.reserve(selected_samples_indxes.size());

        uint progess_cnt = 0;
        int sel_cnt = 0;
#ifdef _OPENMP
 #pragma omp parallel for
#endif
        for(sel_cnt = 0; sel_cnt < selected_samples_indxes.size(); ++sel_cnt)
        {
            string filename = filelist[selected_samples_indxes[sel_cnt]];
            CLASS_LABEL_TYPE label = data_labels[selected_samples_indxes[sel_cnt]];

            Mat training_patch = imread(filename);

            if(training_patch.empty())
            {
                string msg = "Failed to read image file \"" + filename + "\"!";
                throw(FileAccessExecption(msg.c_str()));
            }

            // if training patch has a different size as specified in the ini file
            if(training_patch.size() != Size(_PATCH_WINDOW_SIZE_, _PATCH_WINDOW_SIZE_))
                resize(training_patch, training_patch, Size(_PATCH_WINDOW_SIZE_, _PATCH_WINDOW_SIZE_) );

            vector<float> feature_vec;
            extractFeatures(training_patch, feature_vec);

#ifdef _OPENMP
 #pragma omp critical
#endif
            {
            training_data.push_back(feature_vec);
            training_labels.push_back(label);

            if(++processing_file_msg_cnt >= 71)
            {
                for(int backspace_cnt = 0; backspace_cnt < 512; ++backspace_cnt)
                    cout << '\b';
                cout << "Processing file: " << progess_cnt << " of " << selected_samples_indxes.size() << flush;
                //cout << " (" << filename << ", label: " << label << ")" << flush;
                processing_file_msg_cnt = 0;
            }
            progess_cnt++;
            }
        }
        for(int backspace_cnt = 0; backspace_cnt < 1024; ++backspace_cnt)
            cout << '\b';
        cout << "Feature extraction of training data finished.                                                                     " << endl;

        random_forest_->train(training_data, training_labels);

        // test all samples to determine the training error
        const string test_msg = "Testing all training samples... ";

        processing_file_msg_cnt = 0;
        progess_cnt = 0;
        uint error_cnt = 0;

#ifdef _OPENMP
 #pragma omp parallel for
#endif
        for(int sample_cnt = 0; sample_cnt < nsamples; ++sample_cnt)
        {
            Mat training_patch = imread(filelist[sample_cnt]);
            if(training_patch.empty())
            {
                string msg = "Failed to read image file \"" + filelist[sample_cnt] + "\"!";
                throw(FileAccessExecption(msg.c_str()));
            }

            // Predict the class probability of the training patch
            vector<float> feature_vec;
            extractFeatures(training_patch, feature_vec);

            vector<CRandomForest::WeightedLabel> pred_results;
            random_forest_->predict(feature_vec, pred_results);

            // get classification result
            float highest_object_prob = 0.0f;
            CLASS_LABEL_TYPE class_result = 0;
            for(vector<CRandomForest::WeightedLabel>::const_iterator class_pred_it = pred_results.begin();
                class_pred_it != pred_results.end(); ++class_pred_it)
            {
                // get result class idx and probability
                const float prob = class_pred_it->weight_;
                if(prob > highest_object_prob)
                {
                    class_result = class_pred_it->label_;
                    highest_object_prob = prob;
                }
            }

#ifdef _OPENMP
 #pragma omp critical
#endif
            {
            // if classification result is wrong
            if(class_result != data_labels[sample_cnt])
            {
                error_cnt++;
                hard_sample_idxes.push_back(sample_cnt);
            }

            if(++processing_file_msg_cnt == 191)
            {
                for(int backspace_cnt = 0; backspace_cnt < 1024; ++backspace_cnt)
                    cout << '\b';
                cout << test_msg << "tested file: " << progess_cnt << " of " << nsamples << flush;
                processing_file_msg_cnt = 0;
            }
            progess_cnt++;
            } // omp critical
        }
        train_error = (float)error_cnt/nsamples;

        for(int backspace_cnt = 0; backspace_cnt < 1024; ++backspace_cnt)
            cout << '\b';

        cout << "Bootstrapping-stage " << stages_cnt << " completed: Train error is: " << train_error;
        cout << " (" << error_cnt << " of " << nsamples << " samples, new hard examples size is: " << hard_sample_idxes.size() << ")" << endl;
//        cout << "Hard Examples are: ";
//        for(vector<uint>::const_iterator he_it = hard_sample_idxes.begin(); he_it != hard_sample_idxes.end(); ++he_it)
//        {
//            cout << *he_it << ", ";
//        }
//        cout << endl;
    } // bootstrapping

	// get number of different classes; data_labels is not needed anymore
    vector<CLASS_LABEL_TYPE>::iterator t_it;
	t_it = unique(data_labels.begin(), data_labels.end());
    data_labels.resize(distance(data_labels.begin(), t_it));
    sort(data_labels.begin(), data_labels.end());
    t_it = unique(data_labels.begin(), data_labels.end());
    data_labels.resize(distance(data_labels.begin(), t_it));

    class_labels_ = data_labels;

    //cout << "Number of classes in the training-data: " << distance(data_labels.begin(), t_it) << endl;
}

// for bootstrapping
void CAlgorithmController::bootstrap_getRandomlySelectedSampleIndices(const vector<CLASS_LABEL_TYPE>& data_labels, vector<uint>& selected_sample_indices)
{
    // randomly choose a subset which class distribution is defined by rho+(1-rho)*exp(-(p_o./lambda));
    // rho: max reduction ratio for high class occurance
    // lambda: reduction strength
    const float rho = 0.4;
    const float lambda = 0.3;

    selected_sample_indices.clear();
    const uint nsamples = data_labels.size();

    CLASS_LABEL_TYPE* unique_sorted_data_labels;
    uint nclasses;

    uniqueSortedElements<CLASS_LABEL_TYPE>(data_labels.data(), nsamples, unique_sorted_data_labels, nclasses);

    vector<vector<uint> > class_sample_idxes(nclasses); // all indices ordered by classes
    vector<uint> class_nsamples(nclasses);              // amount of samples ordered by classes
    fill(class_nsamples.begin(), class_nsamples.end(), 0);
    // get for each class label the data indices and the amount of samples
    for(uint smp_cnt = 0; smp_cnt < nsamples; ++smp_cnt)
    {
        for(uint class_cnt = 0; class_cnt < nclasses; ++class_cnt)
        {
            if(unique_sorted_data_labels[class_cnt] == data_labels[smp_cnt])
            {
                class_sample_idxes[class_cnt].push_back(smp_cnt);
                class_nsamples[class_cnt]++;
                break;
            }
        }
    }
    // get the right amount of randomly selected samples for each class
    for(uint class_cnt = 0; class_cnt < nclasses; ++class_cnt)
    {
        // shuffle the indices
        random_shuffle(class_sample_idxes[class_cnt].begin(), class_sample_idxes[class_cnt].end());

        // calculate amount of samples in the selected set according to
        // rho+(1-po)*(1-rho)*exp(-(po/lambda));, where po is the class occurance ratio
        const float p = (float)class_nsamples[class_cnt]/nsamples;
        const float f = rho+(1.0-p)*(1.0-rho)*exp(-(p/lambda)); // rho < f < 1
        const int nselected_samples_for_this_class = floor(f*class_nsamples[class_cnt]);
        // sort the selected samples (they have the same label) for faster HDD access
        sort(class_sample_idxes[class_cnt].begin(), class_sample_idxes[class_cnt].begin()+nselected_samples_for_this_class);
        // now select this amount of samples
        selected_sample_indices.insert(selected_sample_indices.end(), class_sample_idxes[class_cnt].begin(),
                                class_sample_idxes[class_cnt].begin()+nselected_samples_for_this_class);
    }
}


void CAlgorithmController::extractFeatures(const Mat& img_patch, vector<float>& feature_vec)
{
    ELECDETEC_ASSERT(filter_horiz_ && filter_vert_ && hog_extractor_ && gof_extractor_ && haar_extractor_,
                     "Algorithm is not initialized! Features cannot be extracted.");

    vector<float> hog_features, gof_features, color_features, ori_features;

    // extract HoG features
    hog_extractor_->extractFeatureVector(img_patch, hog_features);

    // extract GoF features
    gof_extractor_->extractFeatureVector(img_patch, gof_features);

    // extract orientation features (of the gray image)
    Mat patch_gray;
    cvtColor(img_patch, patch_gray, CV_BGR2GRAY);
    vector<Mat> orientation_gradients;

    Mat grad_vert, grad_horiz;
    filter_horiz_->filterImage(patch_gray, grad_horiz);
    filter_vert_->filterImage(patch_gray, grad_vert);

    orientation_gradients.push_back(grad_horiz);
    orientation_gradients.push_back(grad_vert);

    ori_haar_extractor_->extractFeatureVector(orientation_gradients, ori_features);

    // extract each color channels
//    vector<Mat> color_channels;
//    color_channels.push_back(patch_gray);
//    split(img_patch, color_channels);
    // extract color features
    //haar_extractor_->extractFeatureVector(color_channels, color_features);

    // extract luv channels
    Mat patch_luv = Mat::zeros(img_patch.size(), CV_8UC3);
    // convert to Luv space: does convertion to floating range 0..1 and back to 8-bit range 0..255 for L, u, and v
    cvtColor(img_patch, patch_luv, CV_BGR2Luv);

    vector<Mat> luv_channels;
    split(patch_luv, luv_channels);
    // also add gray channel
    luv_channels.push_back(patch_gray);
    haar_extractor_->extractFeatureVector(luv_channels, color_features);

    feature_vec.clear();
    feature_vec.insert(feature_vec.end(), ori_features.begin(), ori_features.end());
    feature_vec.insert(feature_vec.end(), color_features.begin(), color_features.end());
    feature_vec.insert(feature_vec.end(), hog_features.begin(), hog_features.end());
    feature_vec.insert(feature_vec.end(), gof_features.begin(), gof_features.end());
}

void CAlgorithmController::loadAlgorithmData(const string& filename) throw(FileAccessExecption)
{
    // open xml file
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        string msg = "Failed to read file \"" + filename + "\". Check if file exists and you have read access.";
        throw(FileAccessExecption(msg.c_str()));
    }

    FileNode algo_node = fs[NODE_NAME_ALGORITHM_DATA];

    int swin_size;
    algo_node[NODE_NAME_PATCH_SIZE] >> swin_size;
    if(swin_size != _PATCH_WINDOW_SIZE_)
    {
        string msg = "Failed to load file \"" + filename + "\". Sliding window size is different to the current configuration!";
        throw(FileAccessExecption(msg.c_str()));
    }

    int bglabel;
    algo_node[NODE_NAME_BG_LABEL] >> bglabel;
    if(bglabel != _BACKGROUND_LABEL_)
    {
        string msg = "Failed to load file \"" + filename + "\". Different background label was used at training!";
        throw(FileAccessExecption(msg.c_str()));
    }

    // reset / initialize algorithm parts
    reInitAlgorithm();

    algo_node[NODE_NAME_CLASS_LABELS] >> class_labels_;

    FileNode ori_haar_node = algo_node[NODE_NAME_ORI_HAAR_FEATURES];
    ori_haar_extractor_->load(ori_haar_node);

    FileNode haar_node = algo_node[NODE_NAME_COLOR_HAAR_FEATURES];
    haar_extractor_->load(haar_node);

    FileNode gof_node = algo_node[NODE_NAME_GOF];
    gof_extractor_->load(gof_node);

    FileNode rf_node = algo_node[NODE_NAME_RANDOM_FOREST];
    random_forest_->load(rf_node);

    fs.release();

}

void CAlgorithmController::saveAlgorithmData(const string& filename) const throw(FileAccessExecption)
{
    ELECDETEC_ASSERT( (gof_extractor_ != NULL) && (ori_haar_extractor_!= NULL) && (haar_extractor_ != NULL) && (random_forest_ != NULL),
                      "Trying to save not existing algorithm data!" );

    // create writable file storage object
    FileStorage fs;
    fs.open(filename, FileStorage::WRITE);
    if(!fs.isOpened())
    {
        string msg = "Failed to access/create file \"" + filename + "\". Check write permissions.";
        throw(FileAccessExecption(msg.c_str()));
    }

    fs << NODE_NAME_ALGORITHM_DATA << "{";

    fs << NODE_NAME_PATCH_SIZE << _PATCH_WINDOW_SIZE_;
    fs << NODE_NAME_BG_LABEL << _BACKGROUND_LABEL_;
    fs << NODE_NAME_CLASS_LABELS << class_labels_;
    fs << NODE_NAME_ORI_HAAR_FEATURES << "{"; ori_haar_extractor_->save(fs); fs << "}";
    fs << NODE_NAME_COLOR_HAAR_FEATURES << "{"; haar_extractor_->save(fs); fs << "}";
    fs << NODE_NAME_GOF << "{"; gof_extractor_->save(fs); fs << "}";
    fs << NODE_NAME_RANDOM_FOREST << "{"; random_forest_->save(fs); fs << "}";

    fs << "}"; // NODE_NAME_ALGORITHM_DATA

    fs.release();
}
