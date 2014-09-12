#include <iostream>
#include <vector>
#include <dirent.h>

#include <opencv2/opencv.hpp>

#include "CAnnotation.h"
#include "Utils.h"
#include "tinyxml2.h"
#include "tclap/CmdLine.h"


using namespace std;
using namespace cv;
using namespace tinyxml2;


int main(int argc, char* argv[])
{
	//------------------------------------------------
	// Reading input parameters
	CommandParams c_params;
	parseCmd(argc, argv, c_params);

	//--------------------------------------------
	// Read unprocessed image files
	vector<string> file_list;
	getFileList(c_params.str_dir_, file_list);

	switch(c_params.anno_mode_)
	{
	case ANNOTATION:
	{
		XMLDocument doc;

		// check if there is already a file containing annotations
		if(doc.LoadFile(c_params.str_filename_.c_str()) == XML_SUCCESS)
		{
	    	XMLElement* cur_image;
	    	for(cur_image = doc.FirstChildElement("Annotation"); cur_image != NULL; cur_image = cur_image->NextSiblingElement())
	    	{
	        	string filename = cur_image->Attribute("image");
	        	cout << "Annotation already exists for: " << filename << " skipping.. " << endl;

	        	// if there exists already an annotation for a file that is in the list again, reject it now
	        	vector<string>::iterator finder = find(file_list.begin(), file_list.end(), filename);
	        	if(finder != file_list.end())
	        	{
	        		file_list.erase(finder);
	        	}
	    	}
		}
		else // if an opening error occured, create a new file
		{
			doc.InsertEndChild(doc.NewDeclaration(NULL));
		}

		// manual annotation
		vector<CAnnotation> annotations;
		vector<string>::const_iterator filename_it;
		for(filename_it = file_list.begin(); filename_it != file_list.end(); ++filename_it)
		{
			CAnnotation annotation(*filename_it);
			if(annotation.annotate())
			{
				annotation.addXMLString(doc);
				annotations.push_back(annotation);
			}
			else
			{
				break;
			}
		}

		// save annotations
		doc.SaveFile(c_params.str_filename_.c_str());

		break;
	}
	case REFINEMENT:
	{
		// manually refine the annotated points

		// load existing XML
		XMLDocument doc;
    	doc.Clear();
    	if(doc.LoadFile(c_params.str_filename_.c_str()) != XML_SUCCESS)
    	{
    		cerr << "Loading of XML file failed" << endl;
    		exit(-1);
    	}

    	XMLElement* cur_image_xml;
    	for(cur_image_xml = doc.FirstChildElement("Annotation"); cur_image_xml != NULL; cur_image_xml = cur_image_xml->NextSiblingElement())
    	{
        	string filename = cur_image_xml->Attribute("image");
        	cout << "Annotation: " << filename << endl;
        	CAnnotation cur_image(filename);
        	cur_image.loadFromXMLElement(cur_image_xml);

    	}


    	break;
	}
	case CUTTING:
	{
		// automatic warping, cutting and renaming
		XMLDocument doc;
    	doc.Clear();
    	if(doc.LoadFile(c_params.str_filename_.c_str()) != XML_SUCCESS)
    	{
    		cerr << "Loading of XML file failed" << endl;
    		exit(-1);
    	}

    	MKDIR(c_params.str_dir_.c_str());

    	XMLElement* cur_image_xml;
    	for(cur_image_xml = doc.FirstChildElement("Annotation"); cur_image_xml != NULL; cur_image_xml = cur_image_xml->NextSiblingElement("Annotation"))
    	{
        	string filename = cur_image_xml->Attribute("image");
        	cout << "Annotation: " << filename << endl;
        	CAnnotation cur_image(filename);
        	cur_image.loadFromXMLElement(cur_image_xml);
        	cur_image.saveAnnotatedImages(c_params.str_dir_.c_str());
    	}
    	break;
	}
	}

    //-------------------------------------------

	return 0;
}
