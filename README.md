# hierarchical_clustering_stability

Dataset: https://research.nhgri.nih.gov/microarray/Supplement/ 

Methodology:  https://cran.r-project.org/web/packages/dendextend/vignettes/Cluster_Analysis.html#khan---microarray-gene-expression-data-set-from-khan-et-al.-2001.-subset-of-306-genes.

Tree Distance Metric: normalized Robinson-Foulds distance

Compared Method: Classical Hierarhical Clustering with two different metrics, Info-Clustering with a given metric, a third party method supporting data clustering.

## Steps

### The dataset
`train.csv` and `test.csv`
are extracted from the datastructure of `dendextend::khan$train`
and `dendextend::khan$test` of the package `dendextend`, belonging to R programming language.
