/*
 * SVM.cpp
 *
 *  Created on: Jun 17, 2014
 *      Author: test
 */

#include "SVM.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

CSVM::CSVM(MODULE_CONSTRUCTOR_SIGNATURE)
{
	module_print_name_ = "SVM";
	is_trained_ = false;

	required_input_signature_ = DATA_TYPE_VECTOR | CV_32FC1; // takes float vector
	output_signature_ = DATA_TYPE_WEIGHTED_SCALAR | CV_32FC1;

	if(is_root)
		setAsRoot();

	// SVM Params
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

	auto_train_ = true;

	svm_problem_ = NULL;
	svm_model_ = NULL;
}

CSVM::~CSVM()
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

CVisionData* CSVM::exec()
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


void CSVM::train()
{
	// train_data contains for each sample a row
	CVisionData train_data = getConcatenatedDataAndClearBuffer();

	delete svm_problem_;
	svm_problem_ = new svm_problem();

	cout << "Training SVM. Please be patient..." << flush;

	// prepare training data
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

	//cout << "LinSVM: " << n_samples << " samples copied. Feature size is " << n_features << endl << flush;


	// find best params by parameter grid search
	if(auto_train_)
	{
		cout << " parameter search started... " << flush;
		const int k_fold = 5;

		ValueGrid<double> C_grid(0.5e-1, 1e1, 1.5, ValueGrid<double>::EXP);
		ValueGrid<double> gamma_grid(5e-3, 1e0, 2.5, ValueGrid<double>::EXP);
		ValueGrid<int> degree_grid(1, 10, 1, ValueGrid<int>::LIN);
		ValueGrid<double> coef_grid(1e-2, 1e2, 5.0, ValueGrid<double>::EXP);
		ValueGrid<double> nu_grid(1e-6, 1e1, 3.0, ValueGrid<double>::EXP);
		ValueGrid<double> p_grid(1e-6, 1e1, 3.0, ValueGrid<double>::EXP);
		svm_parameter working_params;
		svm_parameter best_params;
		double best_error = 1.0;

	    // if some parameters are not used by the SVM type and kernel, make their grids static:
	    if( svm_params_->kernel_type != POLY )
	        degree_grid = ValueGrid<int>(svm_params_->degree, svm_params_->degree, 1, ValueGrid<int>::LIN);
	    if( svm_params_->kernel_type == LINEAR )
	        gamma_grid = ValueGrid<double>(svm_params_->gamma, svm_params_->gamma, 1, ValueGrid<double>::LIN);
	    if( svm_params_->kernel_type != POLY && svm_params_->kernel_type != CvSVM::SIGMOID )
	        coef_grid = ValueGrid<double>(svm_params_->coef0, svm_params_->coef0, 1, ValueGrid<double>::LIN);
	    if( svm_params_->svm_type == NU_SVC || svm_params_->svm_type == ONE_CLASS )
	        C_grid = ValueGrid<double>(svm_params_->C, svm_params_->C, 1, ValueGrid<double>::LIN);
	    if( svm_params_->svm_type == C_SVC || svm_params_->svm_type == EPSILON_SVR )
	        nu_grid = ValueGrid<double>(svm_params_->nu, svm_params_->nu, 1, ValueGrid<double>::LIN);
	    if( svm_params_->svm_type != EPSILON_SVR )
	        p_grid = ValueGrid<double>(svm_params_->p, svm_params_->p, 1, ValueGrid<double>::LIN);

		working_params = *svm_params_;

		working_params.C = C_grid.getMin();
		working_params.gamma = gamma_grid.getMin();
		working_params.degree = degree_grid.getMin();
		working_params.coef0 = coef_grid.getMin();
		working_params.nu = nu_grid.getMin();
		working_params.p = p_grid.getMin();

#pragma omp parallel
		for(bool C_reached_lim = false; !C_reached_lim; C_reached_lim = !C_grid.getNextValue(working_params.C))
		{
			for(bool gamma_reached_lim = false; !gamma_reached_lim; gamma_reached_lim = !gamma_grid.getNextValue(working_params.gamma))
			{
				for(bool degree_reached_lim = false; !degree_reached_lim; degree_reached_lim = !degree_grid.getNextValue(working_params.degree))
				{
					for(bool coef_reached_lim = false; !coef_reached_lim; coef_reached_lim = !coef_grid.getNextValue(working_params.coef0))
					{
						for(bool nu_reached_lim = false; !nu_reached_lim; nu_reached_lim = !nu_grid.getNextValue(working_params.nu))
						{
							for(bool p_reached_lim = false; !p_reached_lim; p_reached_lim = !p_grid.getNextValue(working_params.p))
							{
								double* target = Malloc(double, svm_problem_->l);

								const char* check_result = svm_check_parameter(svm_problem_, &working_params);
								if(check_result)
								{
									cerr << "SVM Auto-Training ERROR occurred. Parameter check failed: " << check_result << endl;
									exit(-1);
								}

								svm_cross_validation(svm_problem_, &working_params, k_fold, target);

								int error_cnt = 0;
								for(int i = 0; i < svm_problem_->l; ++i)
								{
									if(target[i] != svm_problem_->y[i])
									{
										++error_cnt;
									}
									//cout << " " << target[i] << svm_problem_->y[i];
								}
								double error = (double)error_cnt / (double)svm_problem_->l;

								if(error < best_error)
								{
									best_params = working_params;
									best_error = error;
								}
							}
						}
					}
				}
			}
		}

		*svm_params_ = best_params;

		cout << endl;

		cout << "Finished auto training with C:" << svm_params_->C <<
				" gamma:" << svm_params_->gamma <<
				" degree:" << svm_params_->degree <<
				" coef0:" << svm_params_->coef0 <<
				" nu:" << svm_params_->nu <<
				" p:" << svm_params_->p <<
				endl;
		cout << "Error is: " << best_error << "(" << best_error*(svm_problem_->l) << " samples)";
	}

	const char* check_result = svm_check_parameter(svm_problem_, svm_params_);
	if(check_result)
	{
		cerr << "SVM Training ERROR occurred: " << check_result << endl;
		exit(-1);
	}

	// train the SVM with svm_params_
	svm_model_ = svm_train(svm_problem_, svm_params_);

	cout << " done. Support Vectors: " << svm_get_nr_sv(svm_model_) << endl << flush;
}

const char* CSVM::svm_type_table_[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

const char* CSVM::kernel_type_table_[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

void CSVM::save(FileStorage& fs) const
{
	cout << "saving SVM..." << endl << flush;

//	svm_save_model("libsvm.txt", svm_model_);

	stringstream config_name;
	config_name << CONFIG_NAME_SVM << "-" << module_id_;

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

	cout << "SVM saved." << endl;
}

void CSVM::load(FileStorage& fs)
{
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
	config_name << CONFIG_NAME_SVM << "-" << module_id_;
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
	cerr << "SVM: Sorry! Loading a sparse SVM from XML not implemented!" << endl;
	exit(-1);

#endif
	svm_model_->free_sv = 1;	// XXX

//	svm_save_model("libsvm_restored.txt", svm_model_);
}
