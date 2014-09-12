/*
 * Utils.h
 *
 *  Created on: Jul 3, 2014
 *      Author: test
 */

#ifndef UTILS_H_
#define UTILS_H_


#include <string>
#include <vector>
#include <dirent.h>


#include "tclap/CmdLine.h"
#include "Exceptions.h"
#include "PipelineController.h"

#if defined _WIN32
  #include <conio.h>
  #include <direct.h>
  #define MKDIR(path) _mkdir(path); // TODO: check if wondows makes directory
  #define FOLDER_CHAR  "\\"
#elif defined __linux__
  #include <sys/types.h>
  #include <sys/stat.h>
  #define MKDIR(path) mkdir(path, 0777); // notice that 777 is different than 0777
  #define FOLDER_CHAR  "/"
#endif

using namespace std;



inline vector<vector<Scalar> > getColors(const int& nclasses)
{
	const int colors_per_class = 1;
	const Scalar_<uchar> color_model(80,240,0);
	vector< vector<Scalar> > ret_colors;
	for(int class_cnt = 0; class_cnt < nclasses; ++class_cnt)
	{
		vector<Scalar> colors;
		for(int color_cnt = 0; color_cnt < colors_per_class; ++color_cnt)
		{
			Scalar color(color_model[2*class_cnt%3], color_model[(2*class_cnt+1)%3], color_model[(2*class_cnt+2)%3]);
			//color[0] = color[0] + color[0] * ((color_cnt - colors_per_class/2)*10);
			colors.push_back(color);
		}
		ret_colors.push_back(colors);
	}
	return ret_colors;
}


struct CommandParams
{
	vector<vector<string> > vec_vec_channels_;
	string str_classifier_, str_imgset_, str_configfile_;
};

inline void parseCmd(int argc, char* argv[], CommandParams& params) throw (ParamExecption)
{
	try {
		// Command Line
		TCLAP::CmdLine cmd("Usage: [-1 IDs [-2 IDs .. -5 IDs]] -f final_classifier -d directory -c filename. For training mode -f must be specified, testing is performed otherwise", ' ', "1.0");

		// Command Arguments
		vector<TCLAP::ValueArg<std::string>*> v_ch_args;
		TCLAP::ValueArg<std::string> c1Arg("1","channel1","IDs of the vision module(s) for the 1st feature channel",false,"","string");
		TCLAP::ValueArg<std::string> c2Arg("2","channel2","IDs of the vision module(s) for the 2nd feature channel",false,"","string");
		TCLAP::ValueArg<std::string> c3Arg("3","channel3","IDs of the vision module(s) for the 3rd feature channel",false,"","string");
		TCLAP::ValueArg<std::string> c4Arg("4","channel4","IDs of the vision module(s) for the 4th feature channel",false,"","string");
		TCLAP::ValueArg<std::string> c5Arg("5","channel5","IDs of the vision module(s) for the 5th feature channel",false,"","string");
		TCLAP::ValueArg<std::string> fArg("f","final","ID of the final classifier vision module. If present, training mode is performed.",false,"","string");
		TCLAP::ValueArg<std::string> dArg("d","dir","Data directory of training- or test-data",true,"","string");
		TCLAP::ValueArg<std::string> cArg("c","config","Configuration file whether the trained pipeline is stored to or loaded from",true,"","string");
		cmd.add( c1Arg ); v_ch_args.push_back( &c1Arg );
		cmd.add( c2Arg ); v_ch_args.push_back( &c2Arg );
		cmd.add( c3Arg ); v_ch_args.push_back( &c3Arg );
		cmd.add( c4Arg ); v_ch_args.push_back( &c4Arg );
		cmd.add( c5Arg ); v_ch_args.push_back( &c5Arg );
		cmd.add( fArg );
		cmd.add( dArg );
		cmd.add( cArg );

		// Command switches
		//TCLAP::SwitchArg trainSwitch("t","train","Train the pipeline", cmd, false);

		// Parse the argv array.
		cmd.parse( argc, argv );

		// Get the values parsed by each arg.

		// for each feature channel
		vector<TCLAP::ValueArg<std::string>*>::iterator ch_arg_ptr_it;
		for(ch_arg_ptr_it = v_ch_args.begin(); ch_arg_ptr_it != v_ch_args.end(); ++ ch_arg_ptr_it)
		{
			TCLAP::ValueArg<std::string>* cur_ch_arg = *ch_arg_ptr_it;
			stringstream ss_args(cur_ch_arg->getValue());

			if(!ss_args.str().empty())
			{
				vector<string> vec_cur_ch_str;
				string item;
				while(getline(ss_args, item, ','))
					vec_cur_ch_str.push_back(item);

				params.vec_vec_channels_.push_back(vec_cur_ch_str);
			}
		}

		params.str_classifier_ = fArg.getValue();
		params.str_imgset_     = dArg.getValue();
		params.str_configfile_ = cArg.getValue();

		// append .xml if missing
		if(params.str_configfile_.compare(params.str_configfile_.size()-4, 4, ".xml"))
			params.str_configfile_ += ".xml";

		// append '/' if missing
		if(params.str_imgset_.compare(params.str_imgset_.size()-1,1,FOLDER_CHAR) != 0)
			params.str_imgset_ += FOLDER_CHAR;

		//train_switch = trainSwitch.getValue();
		std::cout << "Command line:" << endl <<
				     "-------------" << endl;

		for(ch_arg_ptr_it = v_ch_args.begin(); ch_arg_ptr_it != v_ch_args.end(); ++ ch_arg_ptr_it)
		{
			TCLAP::ValueArg<std::string>* cur_ch_arg = *ch_arg_ptr_it;
			if(!cur_ch_arg->getValue().empty())
				cout << " channel " << distance(v_ch_args.begin(), ch_arg_ptr_it) << ": " << cur_ch_arg->getValue() << endl;
		}
		cout <<
				    " final-classifier: " << params.str_classifier_ << endl <<
				    " data-folder:      " << params.str_imgset_ << endl <<
				    " config-file:      " << params.str_configfile_ << endl << endl;
	}
	catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		throw(ParamExecption("parsing failed. call --help to view parameter specifications"));
	}
}


inline bool getFileList(string directory, vector<string>& filelist)
{
	//--------------------------------------------
	// READ IMAGE DIRECTORY FILELIST and filter *.jpg *.png

    if(directory.compare(directory.size()-1,1,"/") != 0)
    	directory += "/";

    filelist.clear();
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.data())) != NULL)
    {
        // print all the files and directories within directory
        while ((ent = readdir(dir)) != NULL)
        {
            std::string filename = directory + ent->d_name;
            if(filename.compare(filename.size()-4,4,".png") == 0 || filename.compare(filename.size()-4,4,".jpg") == 0)
            {
            	filelist.push_back(ent->d_name);
//                std::cout << "File: " << ent->d_name << std::endl << flush;
            }
        }
        closedir(dir);
    }
    else
    {
        /* could not open directory */
        std::cerr << "ERROR: Image directory doesn't exist!" << std::endl;
        return false;
    }

    sort(filelist.begin(), filelist.end());

    return true;
}




#endif /* UTILS_H_ */
