

#include <iostream>
#include <vector>
#include <dirent.h>

#include <opencv2/opencv.hpp>

#include "CAnnotation.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
	// Reading input parameters
	if (argc < 3)
	{
		std::cerr << "Usage: [input-file-dir] [annotation-output-filename]";
		exit(0);
	}
	string str_input_dir = argv[1];
	string output_filename = argv[2];

	vector<string> file_list;
	//--------------------------------------------
	// READ IMAGE DIRECTORY FILELIST and filter *.jpg *.png
    if(str_input_dir.compare(str_input_dir.size()-1,1,"/") != 0)
    	str_input_dir += "/";

    file_list.clear();
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(str_input_dir.data())) != NULL)
    {
        // print all the files and directories within directory
        while ((ent = readdir (dir)) != NULL)
        {
            std::string filename = str_input_dir + ent->d_name;
            if(filename.compare(filename.size()-4,4,".png") == 0 || filename.compare(filename.size()-4,4,".jpg") == 0)
            {
                file_list.push_back(filename);
                std::cout << filename << std::endl;
            }
        }
        closedir(dir);
    }
    else
    {
        /* could not open directory */
        std::cerr << "ERROR: Image directory doesn't exist!" << std::endl;
        exit(-1);
    }
    // END CREATING FILELIST
    //---------------------------------------------


    vector<CAnnotation> annotations;
    vector<string>::const_iterator filename_it;
    for(filename_it = file_list.begin(); filename_it != file_list.end(); ++filename_it)
    {
    	CAnnotation annotation(*filename_it);
    	annotation.annotate();
    	annotations.push_back(annotation);
    }



	return 0;
}
