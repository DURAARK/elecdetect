/*
 * ElecDetec: main.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <map>

#include <Defines.h>
#include <ElecDetec.h>

#include <tinyxml2.h>
#include <simpleini/SimpleIni.h>


using namespace std;
using namespace cv;


void parseCmdLine(int argc, char* argv[], ElecDetec::ExecutionParameter& params, string& inifile) throw(TCLAP::ArgException);
void parseIniFile(const string& filepath) throw(TCLAP::ArgException);

int main(int argc, char* argv[])
{    
    ElecDetec::ExecutionParameter exec_params;
    string inifile;
    try
    {
        parseCmdLine(argc, argv, exec_params, inifile); // throws TCLAP::ArgException
        parseIniFile(inifile);
        ElecDetec detector(exec_params); // throws ElecDetec::ExecParamExecption
        detector.doAction(); // throws ElecDetec::ExecParamExecption
    }
    catch (TCLAP::ArgException& e)
    {
        cerr << "Command line parsing failed (" << e.error() << " for arg " << e.argId() << "). " << endl;
        cerr << "Try --help to view parameter specifications";
        exit(-1);
    }
    catch(ElecDetec::ExecParamExecption& e)
    {
        cerr << e.what() << endl;
        exit(-1);
    }
    return 0;
}


void parseCmdLine(int argc, char* argv[], ElecDetec::ExecutionParameter& params, string& inifile) throw(TCLAP::ArgException)
{
    try {
        // Command Line
        TCLAP::CmdLine cmd("If the given XML file already exists, it is overwritten on training mode. If the some values in the ini-file are undefined, default values are used.", ' ', "1.0");

        // Command Arguments
        TCLAP::ValueArg<std::string> mArg("m","mode","Execution mode, either 'train' or 'detect'.",true,"","string");
        TCLAP::ValueArg<std::string> dArg("d","dir","Data directory of training- or test-data",true,"","string");
        TCLAP::ValueArg<std::string> cArg("c","config","The XML configuration file that is created in training mode and read in detection mode",true,"","string");
        TCLAP::ValueArg<std::string> iArg("i","ini","Configuration file containing several settings for the application. Default is 'config.ini'",false,"config.ini","string");

        cmd.add( mArg );
        cmd.add( dArg );
        cmd.add( cArg );
        cmd.add( iArg );

        // Command switches
        //TCLAP::SwitchArg trainSwitch("t","train","Train the pipeline", cmd, false);

        // Parse the argv array.
        cmd.parse( argc, argv );

        // Get the values parsed by each arg.

        string str_mode        = mArg.getValue();
        params.str_imgset_     = dArg.getValue();
        params.str_configfile_ = cArg.getValue();
        inifile                = iArg.getValue();

        // append .xml if missing
        if(params.str_configfile_.compare(params.str_configfile_.size()-4, 4, ".xml"))
            params.str_configfile_ += ".xml";

        // append '/' if missing
        if(params.str_imgset_.compare(params.str_imgset_.size()-1,1,FOLDER_CHAR) != 0)
            params.str_imgset_ += FOLDER_CHAR;

        //train_switch = trainSwitch.getValue();
        std::cout << "Command line:" << endl <<
                     "-------------" << endl;
        cout <<
                    " mode:             " << str_mode << endl <<
                    " data-folder:      " << params.str_imgset_ << endl <<
                    " XML-file:         " << params.str_configfile_ << endl <<
                    " INI-file:         " << inifile << endl << endl;

        if(str_mode != "train" && str_mode != "detect")
            throw(TCLAP::ArgException("Execution mode not recognized!", mArg.shortID()));

        if(str_mode == "train")
            params.exec_mode_ = ElecDetec::ExecutionParameter::TRAIN;
        if(str_mode == "detect")
            params.exec_mode_ = ElecDetec::ExecutionParameter::DETECT;
    }
    catch (TCLAP::ArgException&)
    {
        // rethrow exception
        throw;
    }
}

// Global Ini-file values

// Detection
string _RESULT_DIRECTORY_NAME_;
bool _WRITE_PROBABILITY_MAPS_;
string _FILENAME_RESULT_SUFFIX_;
string _PROB_MAP_RESULT_SUFFIX_;
float _MAX_BOUNDINGBOX_OVERLAP_;
string _DETECTION_DEFAULT_THRESHOLD_;
string _DETECTION_LABEL_THRESHOLDS_;
string _DETECTION_LABELS_;

// Training
string _LABEL_DELIMITER_;
int _MAX_BOOTSTRAP_STAGES_;

// Common
int _PATCH_WINDOW_SIZE_;
string _IMG_FILE_EXTENTIONS_;
CLASS_LABEL_TYPE _BACKGROUND_LABEL_;


void parseIniFile(const string& filepath) throw(TCLAP::ArgException)
{
    CSimpleIniA inireader;
    inireader.SetUnicode();
    inireader.LoadFile(filepath.c_str());
    if(inireader.IsEmpty())
        throw(TCLAP::ArgException("Ini-File not found!", "-i"));

    _RESULT_DIRECTORY_NAME_ =       inireader.GetValue("detection", "result_directory_name", DEFAULT_RESULT_DIRECTORY_NAME);
    _WRITE_PROBABILITY_MAPS_ =      inireader.GetBoolValue("detection", "write_probability_maps", DEFAULT_WRITE_PROBABILITY_MAPS);
    _FILENAME_RESULT_SUFFIX_ =      inireader.GetValue("detection", "filename_result_suffix", DEFAULT_FILENAME_RESULT_SUFFIX);
    _PROB_MAP_RESULT_SUFFIX_ =      inireader.GetValue("detection", "prob_map_result_suffix", DEFAULT_PROB_MAP_RESULT_SUFFIX);
    _MAX_BOUNDINGBOX_OVERLAP_ =     inireader.GetDoubleValue("detection", "max_boundingbox_overlap", DEFAULT_MAX_BOUNDINGBOX_OVERLAP);

#ifndef PERFORM_EVALUATION
    _DETECTION_DEFAULT_THRESHOLD_ = inireader.GetValue("detection", "detection_default_threshold", DEFAULT_DETECTION_DEFAULT_THRESHOLD);
    _DETECTION_LABEL_THRESHOLDS_ =  inireader.GetValue("detection", "detection_label_thresholds", DEFAULT_DETECTION_LABEL_THRESHOLDS);
    _DETECTION_LABELS_ =            inireader.GetValue("detection", "detection_labels", DEFAULT_DETECTION_LABELS);
#else
    _DETECTION_DEFAULT_THRESHOLD_ = "0.15";
    _DETECTION_LABEL_THRESHOLDS_ = "";
    _DETECTION_LABELS_ = "";
#endif
    _LABEL_DELIMITER_ =      inireader.GetValue("training", "label_delimiter", DEFAULT_LABEL_DELIMITER);
    _MAX_BOOTSTRAP_STAGES_ = inireader.GetLongValue("training", "max_bootstrap_stages", DEFAULT_MAX_BOOTSTRAP_STAGES);

    _PATCH_WINDOW_SIZE_ = inireader.GetLongValue("common", "patch_window_size", DEFAULT_PATCH_SIZE);
    _IMG_FILE_EXTENTIONS_ = inireader.GetValue("common", "file_extentions", DEFAULT_IMG_FILE_EXTENTIONS);
    _BACKGROUND_LABEL_ = inireader.GetLongValue("common", "background_label", DEAFULT_BACKGROUND_LABEL);

//    cout << "Ini-file: " << endl;
//    DEBUG_COUT_VAR(_RESULT_DIRECTORY_NAME_);
//    DEBUG_COUT_VAR(_WRITE_PROBABILITY_MAPS_);
//    DEBUG_COUT_VAR(_FILENAME_RESULT_SUFFIX_);
//    DEBUG_COUT_VAR(_PROB_MAP_RESULT_SUFFIX_);
//    DEBUG_COUT_VAR(_MAX_BOUNDINGBOX_OVERLAP_);
//    DEBUG_COUT_VAR(_DETECTION_DEFAULT_THRESHOLD_);
//    DEBUG_COUT_VAR(_DETECTION_LABEL_THRESHOLDS_);
//    DEBUG_COUT_VAR(_DETECTION_LABELS_);
//    DEBUG_COUT_VAR(_LABEL_DELIMITER_);
//    DEBUG_COUT_VAR(_MAX_BOOTSTRAP_STAGES_);
//    DEBUG_COUT_VAR(_PATCH_WINDOW_SIZE_);
//    DEBUG_COUT_VAR(_IMG_FILE_EXTENTIONS_);
//    DEBUG_COUT_VAR(_BACKGROUND_LABEL_);
//    exit(-1);


}
