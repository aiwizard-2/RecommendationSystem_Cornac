import cornac
# Load MovieLens 100K dataset
ml_100k = cornac.datasets.movielens.load_feedback()

# Split data based on ratio
rs = cornac.eval_methods.RatioSplit(
    data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123
)

# Here we are comparing biased MF, PMF, and BPR
mf = cornac.models.MF(
    k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123
)
pmf = cornac.models.PMF(
    k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123
)
bpr = cornac.models.BPR(
    k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123
)

most_pop = cornac.models.MostPop()

item_knn_bm25 = cornac.models.ItemKNN(
        k=100, similarity="cosine", weighting="bm25", name="ItemKNN-BM25"
    )
bpr = cornac.models.BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
wmf = cornac.models.WMF(
    k=5,
    max_iter=50,
    learning_rate=0.001,
    lambda_u=0.01,
    lambda_v=0.01,
    verbose=True,
    seed=123,
)
vaecf = cornac.models.VAECF(
    k=5,
    autoencoder_structure=[20],
    act_fn="tanh",
    likelihood="mult",
    n_epochs=100,
    batch_size=100,
    learning_rate=0.001,
    beta=1.0,
    seed=123,
    use_gpu=True,
    verbose=True,
)

# Define metrics used to evaluate the models
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=10)
recall = cornac.metrics.Recall(k=10)
ndcg = cornac.metrics.NDCG(k=10)
auc = cornac.metrics.AUC()
mAP = cornac.metrics.MAP()

# Put it together into an experiment and run
cornac.Experiment(
    eval_method=rs,
    # models=[mf, pmf, bpr],
    models=[vaecf],
    metrics=[mae, rmse, prec, recall, ndcg, auc, mAP],
    user_based=True,
).run()