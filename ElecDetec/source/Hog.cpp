#include "Hog.h"


CHog::CHog()
{
	module_name_ = "HoG";

	win_size_ = cv::Size(80, 80); //(128,128)winSize
	cell_size_ = cv::Size(8, 8); //cellSize,
	block_size_ = cv::Size(16, 16);
	block_stride_ = cv::Size(8, 8);
	nbins_ = 9;

	// 9, //nbins,
	// 0, //derivAper,
	// -1, //winSigma,
	// 0, //histogramNormType,
	// 0.2, //L2HysThresh,
	// 0 //gammal correction,
	// //nlevels=64
	//);

	hogy_ = new cv::HOGDescriptor(win_size_, block_size_, block_stride_, cell_size_, nbins_);
}


CHog::~CHog()
{
	delete hogy_;
	hogy_ = NULL;
}

void CHog::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	if(data.back()->getType() != TYPE_MAT)
		throw(VisionDataTypeException(data.back()->getType(), TYPE_MAT));

	const cv::Mat img0 = (((CMat*)data.back())->mat_);

	CVector<float>* hog_features = new CVector<float>();

	cv::Mat img_gray;
	if (img0.channels() != 1)
		cv::cvtColor(img0, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img0;
	cv::resize(img_gray, img_gray, win_size_ );

	hogy_->compute(img_gray, hog_features->vec_);

	//visualize(img_gray, hog_features->vec_);

	data.push_back(hog_features);

}

void CHog::save(FileStorage& fs) const
{

}
void CHog::load(FileStorage& fs)
{

}

int CHog::getFeatureLength()
{
	return hogy_->getDescriptorSize();
}

void CHog::visualize(cv::Mat& origImg, std::vector<float>& descriptorValues)
{
	int scale_factor = 1;
	float viz_factor = 1;

	if (descriptorValues.empty())
	{
		std::cout << "Descriptor is empty! Failed to visualize." << std::endl;
		return;
	}

	cv::Mat visual_image;
	cv::resize(origImg, visual_image, cv::Size(origImg.cols*scale_factor, origImg.rows*scale_factor));
	cv::cvtColor(visual_image, visual_image, cv::COLOR_GRAY2BGR);

	int gradientBinSize = nbins_;
	// dividing 180deg into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14f / (float)gradientBinSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = win_size_.width / cell_size_.width;
	int cells_in_y_dir = win_size_.height / cell_size_.height;
	//int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y < cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x < cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin < gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;

	for (int blockx = 0; blockx < blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky < blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr < 4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}
				for (int bin = 0; bin < gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (int celly = 0; celly < cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin < gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}


	std::cout << "descriptorDataIdx = " << descriptorDataIdx << std::endl;

	// draw cells
	for (int celly = 0; celly < cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cell_size_.width;
			int drawY = celly * cell_size_.height;

			int mx = drawX + cell_size_.width / 2;
			int my = drawY + cell_size_.height / 2;

			rectangle(visual_image,
			cv::Point(drawX*scale_factor, drawY*scale_factor),
			cv::Point((drawX + cell_size_.width)*scale_factor,
				(drawY + cell_size_.height)*scale_factor),
				CV_RGB(100, 100, 100),
				1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin < gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)cell_size_.width / 2;
				float scale = (float)viz_factor; // just a visual_imagealization scale,
				// to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
					cv::Point((int)(x1*scale_factor), (int)(y1*scale_factor)),
					cv::Point((int)(x2*scale_factor), (int)(y2*scale_factor)),
					CV_RGB(0, 0, 255),
					1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)

	cv::imshow("hog-vis", visual_image);
	cv::waitKey(0);
	cv::imwrite("hog-vis.png", visual_image);

	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y < cells_in_y_dir; y++)
	{
		for (int x = 0; x < cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

}
