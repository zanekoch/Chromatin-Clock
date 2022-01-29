# Chromatin-Clock

This is an ElasticNet-based predictor of chronological age from Chromatin State (ChromHMM) files.  

* ./data : would be filled with processed data files, not provided  

./scripts : the preprocessing, model training, and postprocessing scripts  
  -- /preprocessing : does feature selection and combines seperate data files into a single data matrix of features    
    -- -- /auto_correlation_analysis.py : measures the autocorrelation of a chromatin state segmentation to inform feature selection  
    -- -- /average_windows.py : averages adjacent windows in chromatin state segmentation, one option for feature selection  
    -- -- /filter_by_functional_element.py : removes all windows not in a GENCODE annotated functional element, e.g. gene body  
    -- -- /flatten_chromatin_state_files.py : takes all the posterior files for a given chromosome, breaks them into smaller 10k window pieces, and flattens them so that each row is a sample and the columns are the 10k*12 states  
    -- -- /make_feature_matrix.py : given a set of flattened files, raw or partially processed files (e.g. only_gene_bodies), do some feature selection on them and write out the resulting features to a format that ChromatinClock.py can take as input (e.g. .csv file with columns=features, rows=samples)  
 -- /train_model  
    -- -- /ChromatinClock.py : given the file of features chosen (e.g. top variance locaitons) and file of ages of samples, do a hyperparameter search and train an ElasticNet. Also implements 5-fold nested CV  
 -- /post_training_analysis  
    -- -- / test_clock.py : DEPRICATED TESTING NOW DONE IN ChromatinClock.py test the peformance of model trained by ChromatinClock.py by making predictions on testing data  
    -- -- / get_model_features_from_test_data.py : Old method used by test_clock.py, unnecessary now  
    -- -- / feature_analysis.ipynb : exploration of which states are most enriched  
    
