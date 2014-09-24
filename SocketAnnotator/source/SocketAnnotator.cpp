#include <iostream>
#include <vector>
#include <dirent.h>

#include <opencv2/opencv.hpp>

#include "CAnnotation.h"
#include "Utils.h"
#include "tinyxml2.h"


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

	srand(time(NULL));

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
			if(annotation.isImageLoaded() && annotation.annotate())
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
	case REANNOTATION:
	{
		// redefine already annotated points

		// load existing XML
		XMLDocument doc;
    	doc.Clear();
    	if(doc.LoadFile(c_params.str_filename_.c_str()) != XML_SUCCESS)
    	{
    		cerr << "Loading of XML file failed" << endl;
    		exit(-1);
    	}

    	vector<XMLElement*> to_delete;
    	XMLElement* cur_image_xml;
    	for(cur_image_xml = doc.FirstChildElement("Annotation"); cur_image_xml != NULL; cur_image_xml = cur_image_xml->NextSiblingElement("Annotation"))
    	{
        	string filename = cur_image_xml->Attribute("image");
        	cout << "Annotation: " << filename << endl;
        	CAnnotation cur_image(filename);
        	if(cur_image.isImageLoaded())
        	{
        		cur_image.loadFromXMLElement(cur_image_xml);
        		if(cur_image.annotate())
        		{
        			cout << "replacing information" << endl;
        			to_delete.push_back(cur_image_xml);

        			// add the new one (at the end)
        			cur_image.addXMLString(doc);
        		}
        		else
        		{
        			break;
        		}
        	}
    	}

    	// delete old XML Nodes
    	for(vector<XMLElement*>::const_iterator to_del_it = to_delete.begin(); to_del_it != to_delete.end(); ++ to_del_it)
    	{
    		doc.DeleteChild(*to_del_it);
    	}

    	// save annotations
    	doc.SaveFile(c_params.str_filename_.c_str());

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
        	if(cur_image.isImageLoaded())
        	{
        		cur_image.loadFromXMLElement(cur_image_xml);
        		cur_image.saveAnnotatedImages(c_params.str_dir_);
        	}
    	}

    	// writing strictly negative files
    	if(!c_params.str_neg_dir_.empty())
    	{
    		cout << "writing strictly negative samples..." << endl;
    		vector<string> neg_file_list;
    		getFileList(c_params.str_neg_dir_, neg_file_list);
    		for(vector<string>::const_iterator neg_it = neg_file_list.begin(); neg_it != neg_file_list.end(); ++neg_it)
    		{
    			cout << "Negative Image: " << *neg_it << endl;
    			CAnnotation cur_image(*neg_it);
    			if(cur_image.isImageLoaded())
    			{
    				cur_image.saveAnnotatedImages(c_params.str_dir_);
    			}
    		}
    	}

    	break;
	}
	}

    //-------------------------------------------

	return 0;
}
