/*
 * ElecDetec: ElecDetec.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#include "ElecDetec.h"


ElecDetec::ElecDetec(const ExecutionParameter& exec_params) throw(ExecParamExecption) :
    exec_params_(exec_params)
{
}


void ElecDetec::doAction() throw(ExecParamExecption)
{
    // get image file names in directory
    const string imgfolder = exec_params_.str_imgset_; // must have a folder char at the end
    vector<string> imgfiles;
    getImgsFromDir(imgfolder, imgfiles); // can throw ExecParamException

    CAlgorithmController algo;

    switch(exec_params_.exec_mode_)
    {
    case(ExecutionParameter::DETECT): // on testing, load pipeline from configfile
    {
        cout << "Executing DETECT mode for " << imgfolder << endl << flush;
        MKDIR((imgfolder + _RESULT_DIRECTORY_NAME_).c_str());

        for(uint file_cnt = 0; file_cnt < imgfiles.size(); ++file_cnt)
        {
            Mat image = cv::imread(imgfiles[file_cnt]);
            const string image_name = imgfiles[file_cnt];

            vector<Mat> prob_images;
            vector<CLASS_LABEL_TYPE> labels;

            try
            {
                // load pretrained algorithm
                algo.loadAlgorithmData(exec_params_.str_configfile_);
                // get probability images for each class label
                algo.detect(image, image_name, prob_images, labels);
            }
            catch (FileAccessExecption& e)
            {
                cout << e.what() << flush << endl;
                throw(ExecParamExecption("Config file error"));
            }

            // perform post-processing to get concrete detection positions, write xml file and visual output
            postProcessResultImages(image, prob_images, labels, imgfiles[file_cnt], imgfolder);
        }
        break;
    }
    case(ExecutionParameter::TRAIN): // TRAINING mode
    {
        cout << "Executing TRAINING mode for " << imgfolder << endl << flush;
        try
        {
            // train fresh initialized algorithm
            algo.train(imgfiles);

            // save algorithm data
            algo.saveAlgorithmData(exec_params_.str_configfile_);
            cout << exec_params_.str_configfile_ << " saved." << endl;
        }
        catch (FileAccessExecption& e)
        {
            throw(ExecParamExecption(e.what()));
        }
        break;
    }
    }

}


void ElecDetec::postProcessResultImages(const Mat& original_image, const vector<Mat>& prob_imgs, const vector<CLASS_LABEL_TYPE>& labels, const string& input_filename, const string& root_folder)
{
    // create XML Document and make a new image node
    tinyxml2::XMLDocument xml_doc;
    xml_doc.InsertEndChild(xml_doc.NewDeclaration(NULL));
    tinyxml2::XMLElement* xml_image = xml_doc.NewElement("Image");
    xml_image->SetAttribute("file", input_filename.c_str());

    // generate the class-threshold map
    map<CLASS_LABEL_TYPE, float> threshold_map;
    vector<string> ths_str = splitStringByDelimiter(_DETECTION_LABEL_THRESHOLDS_, ",");
    vector<string> ths_lbs_str = splitStringByDelimiter(_DETECTION_LABELS_, ",");
    vector<CLASS_LABEL_TYPE>::const_iterator label_it;
    for(label_it = labels.begin(); label_it != labels.end(); ++label_it)
    {
        // search if a threshold for this label was specified
        stringstream ss;
        vector<string>::const_iterator ths_lbs_str_it;
        for(ths_lbs_str_it = ths_lbs_str.begin(); ths_lbs_str_it != ths_lbs_str.end(); ++ths_lbs_str_it)
        {
            stringstream temp;
            temp << *ths_lbs_str_it;
            CLASS_LABEL_TYPE temp_int;
            temp >> temp_int;
            if(temp_int == *label_it)
            {
                ss << ths_str[distance<vector<string>::const_iterator>(ths_lbs_str.begin(), ths_lbs_str_it)];
            }
        }
        if(ss.str().empty())
            ss << _DETECTION_DEFAULT_THRESHOLD_;

        float threshold;
        ss >> threshold;

        threshold_map[*label_it] = threshold;
    }

    //    for(map<CLASS_LABEL_TYPE, float>::const_iterator map_it = threshold_map.begin(); map_it != threshold_map.end(); ++map_it)
    //    {
    //        cout << "Label: " << map_it->first << " Threshold: " << map_it->second << endl;
    //    }
    //    exit(-1);

    vector<CLabeledWeightedRect> result_bboxes;
    evaluateProbImages(prob_imgs, labels, result_bboxes, threshold_map);

    vector<tinyxml2::XMLElement*> detect_nodes = getDetectionXMLNodes(xml_doc, result_bboxes);
    for(vector<tinyxml2::XMLElement*>::const_iterator dn_it = detect_nodes.begin(); dn_it != detect_nodes.end(); ++dn_it)
        xml_image->InsertEndChild(*dn_it);

    // write files
    // get filename prefix
    vector<string> path_parts = splitStringByDelimiter(input_filename, FOLDER_CHAR);
    vector<string> filename_parts = splitStringByDelimiter(path_parts.back(), ".");
    string filename_prefix;
    for(vector<string>::const_iterator str_part_it = filename_parts.begin(); str_part_it != filename_parts.end() - 1; ++str_part_it)
        filename_prefix += *str_part_it;

    // get input extention (to store the same format)
    vector<string> possible_img_extentions = splitStringByDelimiter(_IMG_FILE_EXTENTIONS_, ",");
    string input_extention;
    for(vector<string>::const_iterator ext_it = possible_img_extentions.begin(); ext_it != possible_img_extentions.end(); ++ext_it)
    {
        if(filename_parts.back() == *ext_it)
        {
            input_extention = *ext_it;
            break;
        }
    }

    // concatenate XML and output image filenames
    string xml_filename = root_folder + _RESULT_DIRECTORY_NAME_ + FOLDER_CHAR + filename_prefix + _FILENAME_RESULT_SUFFIX_ + ".xml";
    string img_filename = root_folder + _RESULT_DIRECTORY_NAME_ + FOLDER_CHAR + filename_prefix + _FILENAME_RESULT_SUFFIX_ + "." + input_extention;

    // write detection results to XML file
    xml_doc.InsertEndChild(xml_image);
    xml_doc.SaveFile(xml_filename.c_str());

    // write images with detection results
    writeResultImage(original_image, result_bboxes, img_filename);

    // write probabililty maps
    if(_WRITE_PROBABILITY_MAPS_)
    {
        vector<Mat>::const_iterator prob_img_it;
        vector<CLASS_LABEL_TYPE>::const_iterator label_it;

        // for each prob image (that is not background)
        for(prob_img_it = prob_imgs.begin(), label_it = labels.begin(); prob_img_it != prob_imgs.end(); ++prob_img_it, ++label_it)
        {
            //PAUSE_AND_SHOW(*prob_img_it)
            // skip background
            if(*label_it == _BACKGROUND_LABEL_)
                continue;

            stringstream label_str;
            label_str << *label_it;

            string prob_img_filename = root_folder + _RESULT_DIRECTORY_NAME_ + FOLDER_CHAR +
                                       filename_prefix + _FILENAME_RESULT_SUFFIX_ +
                                       _PROB_MAP_RESULT_SUFFIX_ + label_str.str() +
                                       "." + input_extention;
            Mat write_img;
            prob_img_it->convertTo(write_img, CV_8UC1, 255);
            // save output image
            imwrite(prob_img_filename, write_img);
        }
    }
}


vector<tinyxml2::XMLElement*> ElecDetec::getDetectionXMLNodes(tinyxml2::XMLDocument& xml_doc, vector<CLabeledWeightedRect>& result_bboxes)
{
    vector<tinyxml2::XMLElement*> detect_nodes;
    for(vector<CLabeledWeightedRect>::const_iterator obj_it = result_bboxes.begin(); obj_it != result_bboxes.end(); ++obj_it)
    {
        tinyxml2::XMLElement* xml_object = xml_doc.NewElement("Object");
        xml_object->SetAttribute("id", (int)distance<vector<CLabeledWeightedRect>::const_iterator>(result_bboxes.begin(), obj_it));

        tinyxml2::XMLElement* xml_label = xml_doc.NewElement("label");
        xml_label->SetText(obj_it->label_);
        xml_object->InsertEndChild(xml_label);

        tinyxml2::XMLElement* xml_weight = xml_doc.NewElement("weight");
        xml_weight->SetText(obj_it->weight_);
        xml_object->InsertEndChild(xml_weight);

        tinyxml2::XMLElement* bounding_box = xml_doc.NewElement("boundingbox");
        bounding_box->SetAttribute("x", obj_it->rect_.x);
        bounding_box->SetAttribute("y", obj_it->rect_.y);
        bounding_box->SetAttribute("w", obj_it->rect_.width);
        bounding_box->SetAttribute("h", obj_it->rect_.height);
        xml_object->InsertEndChild(bounding_box);

        tinyxml2::XMLElement* center_point = xml_doc.NewElement("centerpoint");
        center_point->SetAttribute("x", obj_it->rect_.x + obj_it->rect_.width/2);
        center_point->SetAttribute("y", obj_it->rect_.y + obj_it->rect_.height/2);
        xml_object->InsertEndChild(center_point);

        detect_nodes.push_back(xml_object);
    }
    return detect_nodes;
}


void ElecDetec::nonMaximaSuppression(vector<CLabeledWeightedRect>& candidates, vector<int>& max_indices)
{
    // non-maxima-suppression constant
    const float max_object_overlap = _MAX_BOUNDINGBOX_OVERLAP_;

    max_indices.clear();
    // sort candidates by their weights and start with the highest rated
    // sort rectangles descending in their weights
    sort(candidates.begin(), candidates.end(), CLabeledWeightedRect::greaterWeight);

    vector<CLabeledWeightedRect const*> max_rect_ptrs;

    // ignore overlapping results
    if(!candidates.empty())
    {
        // the rectangle with the highest weight is always the best rated within its region
        max_indices.push_back(0);
        max_rect_ptrs.push_back(&candidates.front());

        // gather other weighted rectanges if they don't overlap with an higher rated
        vector<CLabeledWeightedRect>::const_iterator cand_it;
        for(cand_it = candidates.begin()+1; cand_it != candidates.end(); ++cand_it)
        {
            vector<CLabeledWeightedRect const*>::const_iterator max_rect_it;
            for(max_rect_it = max_rect_ptrs.begin(); max_rect_it != max_rect_ptrs.end(); ++max_rect_it)
                if(cand_it->getOverlapWith(**max_rect_it) > max_object_overlap)
                    break;

            if(max_rect_it == max_rect_ptrs.end()) // no overlapping, better rated rectangle found
            {
                max_rect_ptrs.push_back(&*cand_it);
                max_indices.push_back(distance<vector<CLabeledWeightedRect>::const_iterator>(candidates.begin(), cand_it));
            }
        }
    }
}



void ElecDetec::evaluateProbImages(const vector<Mat>& prob_imgs, const vector<CLASS_LABEL_TYPE>& labels, vector<CLabeledWeightedRect>& result_bboxes, map<CLASS_LABEL_TYPE, float>& threshold_map)
{
    assert(prob_imgs.size() == labels.size());
    result_bboxes.clear();

    // some fixed parameters for mean-shift
    const float ms_kernel_radius = (float)_PATCH_WINDOW_SIZE_/10.0;
    const uint  ms_max_iterations = 20;
    const float ms_epsilon = 0.05;

    // for each probability image, i.e. for each label
    vector<CLabeledWeightedRect> all_label_candidates;
    vector<Mat>::const_iterator prob_img_it;
    vector<CLASS_LABEL_TYPE>::const_iterator label_it;

    // for each prob image (that is not background)
    for(prob_img_it = prob_imgs.begin(), label_it = labels.begin(); prob_img_it != prob_imgs.end(); ++prob_img_it, ++label_it)
    {
        //PAUSE_AND_SHOW(*prob_img_it)
        // skip background
        if(*label_it == _BACKGROUND_LABEL_)
            continue;

        Mat cur_prob_img = prob_img_it->clone();

        // if detection result is binary (non-weighted results) estimate by Gauss-kernel
        if(countNonZero(cur_prob_img >= 0.99) == countNonZero(cur_prob_img > 0.01))
        {
            // first, do opening
            Mat opened_mask = Mat::zeros(cur_prob_img.size(), CV_8UC1);
            morphologyEx(cur_prob_img > 0.5, opened_mask, MORPH_OPEN,
                         getStructuringElement(MORPH_RECT, Size(OPENING_SIZE, OPENING_SIZE) ) ); // Rect due to SW stepsize = OPENING_SIZE

            // apply Gauß-filter to estimate probability
            Mat opened_gauss = Mat::zeros(cur_prob_img.size(), CV_32FC1);
            opened_mask.convertTo(opened_gauss, CV_32FC1, 1.0/255);
            int ksize = 2*static_cast<int>(_PATCH_WINDOW_SIZE_/2)+1;
            GaussianBlur(opened_gauss, opened_gauss, Size(ksize, ksize), (float)_PATCH_WINDOW_SIZE_/15.0);

            opened_gauss.copyTo(cur_prob_img, opened_mask); // mask with opened mask to get sharp boundaries still
        }

        // Perform meanshift for each non-zero-probability to find the peak response
        vector<CLabeledWeightedRect> same_label_candidates;

        // for each point that is not zero
        Mat non_bg_pts;
        findNonZero(cur_prob_img > threshold_map[*label_it], non_bg_pts);
        //PAUSE_AND_SHOW(cur_prob_img)
        for(uint pt_it = 0; pt_it < non_bg_pts.total(); ++pt_it)
        {
            Point seed = non_bg_pts.at<Point>(pt_it);

            Point2f cur_state = seed;
            float inrange_weight_sum = 0;

            // weight is maximum of ms_kernel_radius
            float max_weight_inside_kernel = 0;

            float last_msvec_length = ms_epsilon+1;
            uint ms_iter = 0;
            for(ms_iter = 0; ms_iter < ms_max_iterations && last_msvec_length > ms_epsilon; ++ms_iter)
            {
                // gather (all!) points in range of the Euclidean distance
                vector<Point> points_in_range;
                inrange_weight_sum = 0;
                max_weight_inside_kernel = 0;

                vector<int> x_coords, y_coords;
                linspace<int>(x_coords, floor(cur_state.x-ms_kernel_radius), ceil(cur_state.x+ms_kernel_radius), LINSPACE_DENSE);
                linspace<int>(y_coords, floor(cur_state.y-ms_kernel_radius), ceil(cur_state.y+ms_kernel_radius), LINSPACE_DENSE);

                for(vector<int>::const_iterator x_it = x_coords.begin(); x_it != x_coords.end(); ++x_it)
                    if(*x_it >= 0 && *x_it < cur_prob_img.cols)
                        for(vector<int>::const_iterator y_it = y_coords.begin(); y_it != y_coords.end(); ++y_it)
                            if(*y_it >= 0 && *y_it < cur_prob_img.rows)
                            {
                                const Point candidate_point(*x_it, *y_it);
                                Point2f dist_vec(cur_state.x - candidate_point.x,
                                                 cur_state.y - candidate_point.y);
                                if(norm(dist_vec) <= ms_kernel_radius)
                                {
                                    points_in_range.push_back(candidate_point);
                                    inrange_weight_sum += cur_prob_img.at<float>(candidate_point);
                                    const float pt_weight = cur_prob_img.at<float>(candidate_point);
                                    if(max_weight_inside_kernel < pt_weight)
                                        max_weight_inside_kernel = pt_weight;
                                }
                            }

                // calculate weighted mean of the points within the kernel
                Point2f ms_vector(0.0, 0.0);
                vector<Point>::const_iterator pt_in_range_it;
                for(pt_in_range_it = points_in_range.begin(); pt_in_range_it != points_in_range.end(); ++pt_in_range_it)
                {
                    ms_vector += Point2f((pt_in_range_it->x-cur_state.x) * cur_prob_img.at<float>(*pt_in_range_it) / inrange_weight_sum,
                                         (pt_in_range_it->y-cur_state.y) * cur_prob_img.at<float>(*pt_in_range_it) / inrange_weight_sum);
                }

                last_msvec_length = norm(ms_vector);
                cur_state += ms_vector;
            }

            // Mean-Shift result is now cur_state
            Point2i ms_result(floor(cur_state.x+0.5), floor(cur_state.y+0.5));
            ms_result.x = ms_result.x < 0 ? 0 : ms_result.x;
            ms_result.y = ms_result.y < 0 ? 0 : ms_result.y;
            ms_result.x = ms_result.x > cur_prob_img.cols-1 ? cur_prob_img.cols-1 : ms_result.x;
            ms_result.y = ms_result.y > cur_prob_img.rows-1 ? cur_prob_img.rows-1 : ms_result.y;

            Rect candidate_bb(ms_result.x - _PATCH_WINDOW_SIZE_/2, ms_result.y - _PATCH_WINDOW_SIZE_/2, _PATCH_WINDOW_SIZE_, _PATCH_WINDOW_SIZE_);

#ifdef PERFORM_EVALUATION
            const float weight = max_weight_inside_kernel;
            const float min_percent = threshold_map[*label_it];
#else

            // prob of ms_result peak is maximum weight inside the kernel;
            // use not the original probability from the classifier,
            // but a normalized probability mapping from [THRESHOLD - 100%] -> [MIN_PERCENT - 100%]
            // in order to make object classes with different threshold more comparable
            const float min_percent = 0.5f;
            const float weight = (max_weight_inside_kernel-threshold_map[*label_it])*
                                 ((1.0f-min_percent)/(1.0f-threshold_map[*label_it])) + min_percent;
#endif
            if(weight > min_percent) //  if(weight > threshold_map[*label_it])
                same_label_candidates.push_back(CLabeledWeightedRect(candidate_bb, weight, *label_it));
        }

        // non maxima-suppression for the current label
        vector<int> max_candidate_indices;
        nonMaximaSuppression(same_label_candidates, max_candidate_indices);
        for(vector<int>::const_iterator max_ind_it = max_candidate_indices.begin(); max_ind_it != max_candidate_indices.end(); ++ max_ind_it)
            all_label_candidates.push_back(same_label_candidates[*max_ind_it]);

    } // for each label

    // the final results of all classes
    result_bboxes.clear();

    if(INTER_CLASS_NON_MAXIMA_SUPPRESSION)
    {
        // also eliminate overlapping objects of different labels
        vector<int> max_candidate_indices;
        nonMaximaSuppression(all_label_candidates, max_candidate_indices);
        for(vector<int>::const_iterator max_ind_it = max_candidate_indices.begin(); max_ind_it != max_candidate_indices.end(); ++ max_ind_it)
            result_bboxes.push_back(all_label_candidates[*max_ind_it]);
    }
    else
    {
        // take all detections as they are:
        // sort candidates by their weights and start with the highest rated
        // sort rectangles descending in their weights
        sort(all_label_candidates.begin(), all_label_candidates.end(), CLabeledWeightedRect::greaterWeight);
        result_bboxes = all_label_candidates;
    }

}

//--------------------------------------------
// Draw bounding boxes into an input image and save the output image
void ElecDetec::writeResultImage(const Mat& original_image, const vector<CLabeledWeightedRect>& bb_results, const string& output_filename)
{
    // output image:
    const float overlay_weight = 1;
    Mat result_image = original_image.clone();

    // first: draw bounding boxes
    vector<CLabeledWeightedRect>::const_reverse_iterator result_it;
    for(result_it = bb_results.rbegin(); result_it != bb_results.rend(); ++result_it)
    {
        const Scalar color = getColorByIndex(static_cast<int>(result_it->label_));
        rectangle(result_image, result_it->rect_, color, 4);
        stringstream text_label;
        text_label << "Label " << result_it->label_;
        putText(result_image, text_label.str(), result_it->rect_.tl() + Point(4, 17), FONT_HERSHEY_SIMPLEX, 0.5, color);
        // cout << "Found object bounded by: " << result_it->rect_ << " with label: " << result_it->label_ << " and weight: " << setprecision(4) << result_it->weight_ << endl;
    }
    // then draw probabilities
    for(result_it = bb_results.rbegin(); result_it != bb_results.rend(); ++result_it)
    {
        const Scalar color = getColorByIndex(static_cast<int>(result_it->label_));
        const Scalar gray = Scalar(result_it->weight_*255.0, result_it->weight_*255.0, result_it->weight_*255.0);
        rectangle(result_image, Rect(Point(result_it->rect_.tl().x + _PATCH_WINDOW_SIZE_/2, result_it->rect_.br().y-20), Point(result_it->rect_.br().x-3, result_it->rect_.br().y-3)), gray, -1);
        stringstream text_prob;
        text_prob << setfill('0') << setw(5) << setiosflags(ios::fixed) << setprecision(2) << result_it->weight_*100.0 << "%";
        putText(result_image, text_prob.str(), Point(result_it->rect_.tl().x+_PATCH_WINDOW_SIZE_/2+4, result_it->rect_.br().y-7), FONT_HERSHEY_SIMPLEX, 0.5, color);
    }

    result_image = result_image*overlay_weight + original_image*(1-overlay_weight);
    // save output image
    imwrite(output_filename, result_image);
}
//--------------------------------------------


//--------------------------------------------
// READ IMAGE DIRECTORY FILELIST and filter unsupported files (defined by IMG_FILE_EXTENTIONS)
void ElecDetec::getImgsFromDir(string directory, vector<string>& filelist) throw(ExecParamExecption)
{
    vector<string> possible_img_extentions = splitStringByDelimiter(_IMG_FILE_EXTENTIONS_, ",");

    if(directory.compare(directory.size()-1,1,FOLDER_CHAR) != 0)
        directory += FOLDER_CHAR;

    filelist.clear();
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.data())) != NULL)
    {
        // print all the files and directories within directory
        while ((ent = readdir(dir)) != NULL)
        {
            string filename = directory + ent->d_name;

            // get filename parts
            vector<string> filename_parts = splitStringByDelimiter(filename, ".");

            // search for supported file extention (the last filename part)
            for(vector<string>::const_iterator ext_it = possible_img_extentions.begin(); ext_it != possible_img_extentions.end(); ++ext_it)
            {
                if(filename_parts.back() == *ext_it)
                {
                    filelist.push_back(directory + string(ent->d_name));
                    break;
                }
            }
        }
        closedir(dir);
    }
    else
    {
        /* could not open directory */
        throw ExecParamExecption("Specified image directory doesn't exist!");
    }
    sort(filelist.begin(), filelist.end());
}
//--------------------------------------------

