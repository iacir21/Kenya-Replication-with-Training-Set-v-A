# Kenya-Replication-with-Training-Set-v-A

1) Train and test sets are seperated. Median year by the number of cases for each judge has been used. Median year is included in train set.

2) The number of tokens greater than 50000 for each jusge is the treshold (same as the previous implementation) There might be a discrepancy between the current and previous implementations. I excluded all the judges who do not have any slant scores in the merged_metadata_v15.csv Also, normalized name is problematic for me to implement the same list of judges from the previous code. (As I don't have df_thresh.csv file)

Below two steps are Stanford Implementation
3) judge_corpus creates bootstrapped corpus for each judge. Encoding can be controlled. But It seems fine. 
4) trains the embeddings (same as stanford implementtaion) 
