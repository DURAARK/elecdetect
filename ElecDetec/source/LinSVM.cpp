/*
 * CLinSVM.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: test
 */

#include "LinSVM.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

CLinSVM::CLinSVM(MODULE_CONSTRUCTOR_SIGNATURE)
{
	module_print_name_ = "LinSVM";
	is_trained_ = false;

	required_input_signature_ = DATA_TYPE_VECTOR | CV_32FC1; // takes float vector
	output_signature_ = DATA_TYPE_WEIGHTED_SCALAR | CV_32FC1;

	if(is_root)
		setAsRoot();

	// SVM Params
	// int svm_type, int kernel_type, double degree, double gamma, double coef0, double Cvalue, double nu, double p, CvMat* class_weights, CvTermCriteria term_crit
	// default:     svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0), gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0)

	//defaults:
	int svm_type = C_SVC;
	int kernel_type = RBF; // LINEAR;
	int degree = 0;	/* for poly */
	double gamma = 3.3750000000000002e-02;//1.0;	/* for poly/rbf/sigmoid */
	double coef0 = 0.0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size = 512; /* in MB */
	double eps = 1.2e-7;//0.001;	/* stopping criteria */
	double C = 2.5;//1.0;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight = 0;		/* for C_SVC */
	int *weight_label = NULL;	/* for C_SVC */
	double* weight = NULL;		/* for C_SVC */
	double nu = 0.1;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p = 0.0;	/* for EPSILON_SVR */
	int shrinking = 0;	/* use the shrinking heuristics */
	int probability = 0; /* do probability estimates */

	svm_params_ = new svm_parameter();
//	svm_params_ = new svm_parameter(svm_type, kernel_type, degree, gamma, coef0, cache_size, eps,
//			            C, nr_weight, weight_label, weight, nu, p, shrinking, probability);
//	svm_parameter sp = {svm_type, kernel_type, degree, gamma, coef0, cache_size, eps,
//            C, nr_weight, weight_label, weight, nu, p, shrinking, probability};
	svm_params_->svm_type = svm_type;
	svm_params_->kernel_type = kernel_type;
	svm_params_->degree = degree;
	svm_params_->gamma = gamma;
	svm_params_->coef0 = coef0;
	svm_params_->cache_size = cache_size;
	svm_params_->eps = eps;
	svm_params_->C = C;
	svm_params_->nr_weight = nr_weight;
	svm_params_->weight_label = weight_label;
	svm_params_->weight = weight;
	svm_params_->nu = nu;
	svm_params_->p = p;
	svm_params_->shrinking = shrinking;
	svm_params_->probability = probability;

	svm_problem_ = NULL;
	svm_model_ = NULL;

}

CLinSVM::~CLinSVM()
{
	if(svm_model_)
	{
		svm_free_and_destroy_model(&svm_model_);
		//delete svm_model_;
	}
	svm_model_ = NULL;

	if(svm_params_)
	{
		svm_destroy_param(svm_params_);
	}
	svm_params_ = NULL;

	// TODO: free svm-problem!!!
	if(svm_problem_)
		delete svm_problem_;
	svm_problem_ = NULL;
}

CVisionData* CLinSVM::exec()
{
	// data-buffer already stores the converted data
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	Mat result_scalar = Mat::zeros(1,2,CV_32FC1); // First Value: Label, second: weight

	svm_node* temp_node = Malloc(svm_node, 1);
	temp_node->values = Malloc(double, working_data.data().cols);
	for(int i = 0; i < working_data.data().cols; i++)
		temp_node->values[i] = working_data.data().at<float>(i);
	temp_node->dim = working_data.data().cols;
	//double* prob_est = Malloc(double, svm_get_nr_class(svm_model_));

	result_scalar.at<float>(0,0) = static_cast<float>(svm_predict(svm_model_, temp_node)) - 1;
	result_scalar.at<float>(0,1) = static_cast<float>(1.0);
//	if(result_scalar.at<float>(0,0) != 0)
//		cout << "prediction result is: " << result_scalar.at<float>(0,0) << " with probability: " << result_scalar.at<float>(0,1) << endl;

	//free(prob_est);
	free(temp_node->values);
	free(temp_node);

	return new CVisionData(result_scalar, DATA_TYPE_WEIGHTED_SCALAR);
}


void CLinSVM::train()
{
	// train_data contains for each sample a row
	//assert(train_data.data().rows == static_cast<int>(train_labels.data().rows));
	CVisionData train_data = getConcatenatedDataAndClearBuffer();

	delete svm_problem_;
	svm_problem_ = new svm_problem();

	//const int k_fold = 5;


	const int n_samples = train_data.data().rows;
	const int n_features = train_data.data().cols;


	double* problem_y = Malloc(double, n_samples);//(double*)malloc(n_samples*sizeof(double));
	svm_node* problem_x = Malloc(svm_node, n_samples);//(svm_node*)malloc(n_samples*sizeof(struct svm_node));


	svm_problem_->l = n_samples; // data length
	// problem.y = int* data labels (beginning from 1)
	// SPARSE: problem.x = svm_node**, x[sample_nr] = [ (f_index, value), (f_index,value), ...] ; f_index must be ASCENDING


	for(int sample_cnt = 0; sample_cnt < n_samples; ++sample_cnt)
	{
		problem_y[sample_cnt] = static_cast<double>(data_labels_->data().at<int>(sample_cnt)) + 1; // libsvm indices start from 1!

		problem_x[sample_cnt].dim = n_features;
		problem_x[sample_cnt].values = Malloc(double, n_features);//(double*)malloc(n_features*sizeof(double));
		for(int feature_cnt = 0; feature_cnt < n_features; ++feature_cnt)
		{
			problem_x[sample_cnt].values[feature_cnt] = static_cast<double>(train_data.data().at<float>(sample_cnt, feature_cnt));
		}
	}

	svm_problem_->y = problem_y;
	svm_problem_->x = problem_x;


	//cv::Mat train_data_mat = train_data.mat_; // generate cvMat without copying the data. need CV_32FC1 cv::Mat as train data
	//cv::Mat train_labels_mat(train_labels.vec_, false); // generate cvMat without copying the data. need CV_32SC1 as train labels

//	cout << "Matrix: " << train_data_mat.rows << "x" << train_data_mat.cols << endl;
//	cout << "first value" << train_data_mat.at<float>(0,0) << endl;

//	double min, max;
//	minMaxLoc(train_data.data(), &min, &max);
//	cout << "min-max: " << min << " - " << max << endl << flush;
//	cout << "Data Labels: " << endl << data_labels_->data().t() << endl;

//	// constructor for matrix headers pointing to user-allocated data
//    Mat(int _rows, int _cols, int _type, void* _data, size_t _step=AUTO_STEP);
//    Mat(Size _size, int _type, void* _data, size_t _step=AUTO_STEP);

//	cout << "Type of Train Data Mat: " << train_data.type2str() << endl;
//	cout << " with size: " << train_data_mat.rows << " x " << train_data_mat.cols << endl;
//	cout << "Type of Train Label Mat: " << train_labels_mat.type() << " should be " << CV_32SC1 << endl;
//	cout << " with size: " << train_labels_mat.rows << " x " << train_labels_mat.cols << endl;

//	cout << flush;

	const char* check_result = svm_check_parameter(svm_problem_, svm_params_);
	if(check_result)
	{
		cerr << "SVM Training ERROR occurred: " << check_result << endl;
		exit(-1);
	}

	cout << "Training SVM. Please be patient..." << flush;

	svm_model_ = svm_train(svm_problem_, svm_params_);



	//svm_->train(train_data_mat, train_labels_mat, Mat(), Mat(), *svm_params_);
	//svm_->train_auto(train_data.data(), data_labels_->data(), Mat(), Mat(), *svm_params_, 5);

	cout << " done. Support Vectors: " << svm_get_nr_sv(svm_model_) << endl << flush;
}

const char* CLinSVM::svm_type_table_[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

const char* CLinSVM::kernel_type_table_[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

void CLinSVM::save(FileStorage& fs) const
{
	cout << "saving LinSVM..." << endl << flush;

	svm_save_model("libsvm.txt", svm_model_);

	stringstream config_name;
	config_name << CONFIG_NAME_LIN_SVM << "-" << module_id_;

    fs << config_name.str().c_str() << "{";

	const svm_parameter& param = svm_model_->param;

	fs << "svm_type" << svm_type_table_[param.svm_type];
	fs << "kernel_type" << kernel_type_table_[param.kernel_type];

	if(param.kernel_type == POLY)
		fs << "degree" << param.degree;

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fs << "gamma" << param.gamma;

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fs << "coef0" << param.coef0;

	int nr_class = svm_model_->nr_class;
	int l = svm_model_->l;
	fs << "nr_class" << nr_class;
	fs << "total_sv" << l;


	{
		fs << "rho" << "[:";
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fs << svm_model_->rho[i];
		fs << "]";
	}

	if(svm_model_->label)
	{
		fs << "label" << "[:";
		for(int i=0;i<nr_class;i++)
			fs << svm_model_->label[i];
		fs << "]";
	}

	if(svm_model_->probA) // regression has probA only
	{
		fs << "probA" << "[:";
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fs << svm_model_->probA[i];
		fs << "]";
	}
	if(svm_model_->probB)
	{
		fs << "probB" << "[:";
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fs << svm_model_->probB[i];
		fs << "]";
	}

	if(svm_model_->nSV)
	{
		fs << "nr_sv" << "[:";
		for(int i=0;i<nr_class;i++)
			fs << svm_model_->nSV[i];
		fs << "]";
	}

	fs << "SV" << "{"; //fprintf(fp, "SV\n");

	const double * const *sv_coef = svm_model_->sv_coef;
#ifdef _DENSE_REP
	const svm_node *SV = svm_model_->SV;
#else
	const svm_node * const *SV = svm_model_->SV;
#endif

	for(int i=0;i<l;i++)
	{
		stringstream nr_i;
		nr_i << "SV" << i;
		fs << nr_i.str().c_str() << "{";
		fs << "coefs" << "[:";
		for(int j=0;j<nr_class-1;j++)
			fs << sv_coef[j][i];// fprintf(fp, "%.16g ",sv_coef[j][i]);
		fs << "]";

#ifdef _DENSE_REP
		const svm_node *p = (SV + i);

		fs << "values" << "[:";

		if(param.kernel_type == PRECOMPUTED)
			fs << (int)p->values[0]; //fprintf(fp,"0:%d ",(int)(p->values[0]));
		else
			for (int j = 0; j < p->dim; j++)
				//if (p->values[j] != 0.0)
				{
//					stringstream nr;
//					nr << j;
					fs << p->values[j]; //fprintf(fp,"%d:%.8g ",j, p->values[j]);
				}
		fs << "]";
#else
		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fs << (int)p->value; //fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				stringstream index;
				index << "val" << p->index;
				fs << index.str().c_str() << p->value; //fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
#endif

		fs << "}"; //fprintf(fp, "\n");
	}



	fs << "}" << "}";



	cout << "LinSVM saved." << endl;
}

void CLinSVM::load(FileStorage& fs)
{
//	if(svm_)
//	{
//		stringstream config_name;
//		config_name << CONFIG_NAME_SVM << "-" << module_id_;
//		svm_->clear();
//		svm_->read(*fs, cvGetFileNodeByName(*fs, NULL, config_name.str().c_str()));
//	}

	if(svm_model_)
	{
		svm_free_and_destroy_model(&svm_model_);
		//delete svm_model_;
	}

	svm_model_ = Malloc(svm_model, 1);
	svm_model_->rho = NULL;
	svm_model_->probA = NULL;
	svm_model_->probB = NULL;
	svm_model_->sv_indices = NULL;
	svm_model_->label = NULL;
	svm_model_->nSV = NULL;

	stringstream config_name;
	config_name << CONFIG_NAME_LIN_SVM << "-" << module_id_;
	FileNode fn = fs[config_name.str().c_str()];

	// BEGIN read header:
	svm_parameter& param = svm_model_->param;

	// - svm type
	string str_svm_type;
	fn["svm_type"] >> str_svm_type;
	int i;
	for(i=0; svm_type_table_[i]; i++)
	{
		if(strcmp(svm_type_table_[i],str_svm_type.c_str()) == 0)
		{
			param.svm_type=i;
			break;
		}
	}
	if(svm_type_table_[i] == NULL)
	{
		cerr << "LinSVM load ERROR: unknown svm type." << endl;
		exit(-1);
	}

	// - kernel type
	string str_kernel_type;
	fn["kernel_type"] >> str_kernel_type;

	for(i=0; kernel_type_table_[i]; i++)
	{
		if(strcmp(kernel_type_table_[i],str_kernel_type.c_str()) == 0)
		{
			param.kernel_type=i;
			break;
		}
	}
	if(kernel_type_table_[i] == NULL)
	{
		cerr << "LinSVM load ERROR: unknown kernel function." << endl;
		exit(-1);
	}

	fn["degree"] >> param.degree;
	fn["gamma"] >> param.gamma;
	fn["coef0"] >> param.coef0;
	fn["nr_class"] >> svm_model_->nr_class;
	fn["total_sv"] >> svm_model_->l;

	// - rhos
	{
		vector<double> temp_vec;
		fn["rho"] >> temp_vec;
		if(!temp_vec.empty())
		{
			int n = svm_model_->nr_class * (svm_model_->nr_class-1)/2;
			svm_model_->rho = Malloc(double,n);
			for(int i = 0; i < n; i++)
				svm_model_->rho[i] = temp_vec[i];
		}
	}

	// - labels
	{
		vector<int> temp_vec;
		fn["label"] >> temp_vec;
		if(!temp_vec.empty())
		{
			int n = svm_model_->nr_class;
			svm_model_->label = Malloc(int,n);
			for(int i = 0; i < n; i++)
				svm_model_->label[i] = temp_vec[i];
		}
	}

	// - probA
	{
		vector<double> temp_vec;
		fn["probA"] >> temp_vec;
		if(!temp_vec.empty())
		{
			int n = svm_model_->nr_class * (svm_model_->nr_class-1)/2;
			svm_model_->probA = Malloc(double, n);
			for(int i = 0; i < n; i++)
				svm_model_->probA[i] = temp_vec[i];
		}
	}

	// - probB
	{
		vector<double> temp_vec;
		fn["probB"] >> temp_vec;
		if(!temp_vec.empty())
		{
			int n = svm_model_->nr_class * (svm_model_->nr_class-1)/2;
			svm_model_->probB = Malloc(double, n);
			for(int i = 0; i < n; i++)
				svm_model_->probB[i] = temp_vec[i];
		}
	}

	// - nr_sv
	{
		vector<int> temp_vec;
		fn["nr_sv"] >> temp_vec;
		if(!temp_vec.empty())
		{
			int n = svm_model_->nr_class;
			svm_model_->nSV = Malloc(int, n);
			for(int i = 0; i < n; i++)
				svm_model_->nSV[i] = temp_vec[i];
		}
	}
	// END read header


	// read sv_coef and SV
	int m = svm_model_->nr_class - 1;
	int l = svm_model_->l;
	svm_model_->sv_coef = Malloc(double *,m);

	for(i=0; i<m; i++)
		svm_model_->sv_coef[i] = Malloc(double,l);

#ifdef _DENSE_REP
	svm_model_->SV = Malloc(svm_node,l);

	// for each Support Vector
	for(i = 0; i < l; i++)
	{
		// get current support vector file node
		stringstream cur_sv_name;
		cur_sv_name << "SV" << i;
		FileNode sv_node = fn["SV"][cur_sv_name.str().c_str()];

		if(sv_node.empty())
			cout << cur_sv_name.str() << " is EMPTY" << endl << flush;

		// read SV coefs
		vector<double> temp_coefs;
		sv_node["coefs"] >> temp_coefs;
		for(int k = 0; k < m; k++)
			svm_model_->sv_coef[k][i] = temp_coefs[k];

		// read SV's
		vector<double> temp_values;
		sv_node["values"] >> temp_values;
		svm_model_->SV[i].values = Malloc(double, temp_values.size());
		svm_model_->SV[i].dim = static_cast<int>(temp_values.size());

		for(int k = 0; k < static_cast<int>(temp_values.size()); k++)
			svm_model_->SV[i].values[k] = temp_values[k];
	}

#else
	cerr << "LinSVM: Sorry! Loading a sparse SVM from XML not implemented!" << endl;
	exit(-1);

#endif


	svm_model_->free_sv = 1;	// XXX

	svm_save_model("libsvm_restored.txt", svm_model_);

//	cout << "SVM load: Support Vectors: " << svm_->get_support_vector_count() << endl;
//	cout << *svm_->get_support_vector(0) << ", " << *svm_->get_support_vector(1) << ", " << *svm_->get_support_vector(2) << endl;
}




