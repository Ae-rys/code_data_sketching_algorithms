This is the code from my L3 internship : "Efficient sketching for frequency estimation for heavy-tail distributions".

#Presentation of the files:

  ##How I generated the graphs in the report:\
    - count_min_vs_count.py contains the code for my comparison between count min and count sketch
    - mle_estimator.py contains the code for my MLE estimator for a Zipfian distribution\
    - hot_and_cold.py contains the code from the Hot & Cold analysis

  ##Other files:\
    - experiment_SBF_CM.py contains the code I used to compare the performance of the Spectral bloom filter and of the Count Min\
    - sketches_one_line.py contains the same code as count_min_vs_count.py, but the Count Min and the Count Sketch use a unique table instead of a matrix (hence, this "Count Min Sketch" is in fact a Spectral Bloom Filter)

