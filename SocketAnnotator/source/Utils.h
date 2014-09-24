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

using namespace std;
using namespace cv;


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


#define PAUSE_AND_SHOW(image) \
    cout << "  Paused - press any key" << endl; \
    namedWindow("DEBUG"); \
    imshow("DEBUG", image); \
    waitKey(); \
    destroyWindow("DEBUG");



enum AnnoMode {ANNOTATION, REANNOTATION, CUTTING};

struct CommandParams
{
	string str_dir_, str_filename_, str_neg_dir_;
	AnnoMode anno_mode_;
};

inline void parseCmd(int argc, char* argv[], CommandParams& params)
{
	try {
		// Command Line
		TCLAP::CmdLine cmd("Usage: SocketAnnotator -d [directory] -m [mode] -f [filename] [-n directory]", ' ', "1.0");

		// Command Arguments
		TCLAP::ValueArg<std::string> mArg("m","mode","Annotation Mode: either 'anno', 'reanno', or 'cut'",true,"","string");
		TCLAP::ValueArg<std::string> dArg("d","dir","Directory of the unprocessed input images",true,"","string");
		TCLAP::ValueArg<std::string> fArg("f","file","Annotation file which should be generated or to be read from (depends on the mode)",true,"","string");
		TCLAP::ValueArg<std::string> nArg("n","neg","Directory where additional strictly negative images are stored (optional)",false,"","string");

		cmd.add( mArg );
		cmd.add( dArg );
		cmd.add( fArg );
		cmd.add( nArg );

		// Command switches
		//TCLAP::SwitchArg trainSwitch("t","train","Train the pipeline", cmd, false);

		// Parse the argv array.
		cmd.parse( argc, argv );

		// Get the values parsed by each arg.

		params.str_neg_dir_ = nArg.getValue();
		params.str_dir_ = dArg.getValue();
		params.str_filename_  = fArg.getValue();
		string anno_mode;
		if(mArg.getValue() == "anno")
		{
			params.anno_mode_ = ANNOTATION;
			anno_mode = "Annotation";
		}
		else if(mArg.getValue() == "reanno")
		{
			params.anno_mode_ = REANNOTATION;
			anno_mode = "Re-Annotation";
		}
		else if(mArg.getValue() == "cut")
		{
			params.anno_mode_ = CUTTING;
			anno_mode = "Cutting";
		}
		else
		{
			cerr << "Mode string not recognized" << endl;
			exit(-1);
		}

		// append .xml if missing
		if(params.str_filename_.compare(params.str_filename_.size()-4, 4, ".xml"))
			params.str_filename_ += ".xml";

		// append '/' if missing
		if(params.str_dir_.compare(params.str_dir_.size()-1,1,FOLDER_CHAR) != 0)
			params.str_dir_ += FOLDER_CHAR;

		//train_switch = trainSwitch.getValue();
		std::cout << "Command line:" << endl <<
				     "-------------" << endl;

		cout <<
				    " filename:        " << params.str_filename_  << endl <<
				    " image-folder:    " << params.str_dir_ << endl <<
				    " mode:            " << anno_mode             << endl << endl;
	}
	catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		exit(-1);
	}
}


inline bool getRecursiveDirContent(const string& directory, vector<string>& filelist)
{
    DIR *dir;
    struct dirent *entry;

    if (!(dir = opendir(directory.c_str())))
    {
        /* could not open directory */
        std::cerr << "ERROR: Image directory doesn't exist!" << std::endl;
        return false;
    }
    if (!(entry = readdir(dir)))
    {
        /* could not open directory */
        std::cerr << "ERROR: Image directory doesn't exist!" << std::endl;
        return false;
    }

    do {
        if (entry->d_type == DT_DIR)
        {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;

        	string path = directory + entry->d_name;
            if(path.compare(path.size()-1,1,"/") != 0)
            	path += "/";
            //filelist.push_back(entry->d_name); //printf("%*s[%s]\n", level*2, "", entry->d_name);
            getRecursiveDirContent(path, filelist);
        }
        else
        {
        	string adder;
            if(directory.compare(directory.size()-1,1,"/") != 0)
            	adder += "/";
        	filelist.push_back(directory + adder + entry->d_name);  //printf("%*s- %s\n", level*2, "", entry->d_name)
        }
    } while ((entry = readdir(dir)) != NULL);
    closedir(dir);

    return true;
}

inline bool getFileList(string directory, vector<string>& filelist)
{
	//--------------------------------------------
	// READ IMAGE DIRECTORY FILELIST and filter *.jpg *.png

    if(directory.compare(directory.size()-1,1,"/") != 0)
    	directory += "/";

    filelist.clear();

    vector<string> full_filelist;

    if(getRecursiveDirContent(directory.c_str(), full_filelist))
    {
    	for(vector<string>::const_iterator f_it = full_filelist.begin(); f_it != full_filelist.end(); ++f_it)
    	{
    		if(f_it->compare(f_it->size()-4,4,".png") == 0 || f_it->compare(f_it->size()-4,4,".jpg") == 0 ||
    				f_it->compare(f_it->size()-4,4,".PNG") == 0 || f_it->compare(f_it->size()-4,4,".JPG") == 0)
    		{
    			//std::cout << "File: " << *f_it << std::endl << flush;
    			filelist.push_back(*f_it);
    		}
    	}
    }

//    DIR *dir;
//    struct dirent *ent;
//    if ((dir = opendir(directory.c_str())) != NULL)
//    {
//        // print all the files and directories within directory
//        while ((ent = readdir(dir)) != NULL)
//        {
//            std::string filename = directory + ent->d_name;
//            std::cout << "File: " << ent->d_name << std::endl << flush;
//            if(filename.compare(filename.size()-4,4,".png") == 0 || filename.compare(filename.size()-4,4,".jpg") == 0 ||
//               filename.compare(filename.size()-4,4,".PNG") == 0 || filename.compare(filename.size()-4,4,".JPG") == 0)
//            {
//            	filelist.push_back(directory + ent->d_name);
////                std::cout << "File: " << ent->d_name << std::endl << flush;
//            }
//        }
//        closedir(dir);
//    }
//    else
//    {
//        /* could not open directory */
//        std::cerr << "ERROR: Image directory doesn't exist!" << std::endl;
//        return false;
//    }

    sort(filelist.begin(), filelist.end());

    return true;
}




#endif /* UTILS_H_ */
