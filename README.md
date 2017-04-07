# Hierarchical Recurrent Encoder-Decoder code (HRED) for Query Suggestion.

This code accompanies the paper:

"A Hierarchical Recurrent Encoder-Decoder For Generative Context-Aware Query Suggestion", by Alessandro Sordoni, Yoshua Bengio, Hossein Vahabi, Christina Lioma, Jakob G. Simonsen, Jian-Yun Nie, to appear in CIKM'15.

The pre-print of the paper is available at: http://arxiv.org/abs/1507.02221.

-- Data processing

The dataset must consist in two files:

data.ses: each line is a sequence of tab-separated strings (queries). Each line represents a query session.
data.rnk: each line is a sequence of tab-separated integers (not currently used in the model, can be set to a tab-separated list of 0).

Basically, the .rnk file is not used by the model but it contains the rank of the clicked documents for each of the queries.

./convert-text2dict.py data

This will create the preprocessed dataset for training.

-- Training

Create a prototype by modifying state.py and launch:

python train.py --prototype your_prototype
