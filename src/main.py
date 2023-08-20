import torch
import pytorch_lightning as pl

from torch.nn import BCEWithLogitsLoss
from label_noise.clean_labels_helpers import *
from pulearn.neural_nets.data_modules import TextDataModule
from pulearn.neural_nets.estimators import PUNet, CnnEstimator
from pulearn.loss_functions.ramp_loss import RampLossNNPU
from pulearn.loss_functions.nnpu_loss import NNPULoss
from textprep.embed import EmbeddingLoader
from copy import copy
from logger import *
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from configuration import *


# Driver code
if __name__ == "__main__":
    # Load configuration settings
    settings = load_settings("app-settings.json")

    # Create a download manager that handles
    # downloading the pretrained embeddings, datasets, etc ...
    download_mgr = DownloadManager(settings)

    experiment = {"Dataset": settings["datamodule_params"]["csv_file"]}

    # The text data module is the main
    # component that prepares and provides
    # data to the deep learning model which will be used
    params = settings["datamodule_params"]
    datamodule = TextDataModule(
        download_mgr=download_mgr,
        **params
    )

    experiment["init_train_split"] = datamodule.train_split_logs
    experiment["init_val_split"] = datamodule.val_split_logs
    experiment["init_test_split"] = datamodule.test_split_logs

    # The embedding loader currently supports 
    # loading GloVe & FastText pretrained embeddings
    loader = EmbeddingLoader(
        datamodule.vectorizer.text_vocab,
        mgr=download_mgr
    )

    embeddings_option = settings["pretrained_embedding"]["option"]
    dim = settings["pretrained_embedding"]["dim"]

    experiment["option"] = embeddings_option
    experiment["dim"] = dim
    
    pretrained = loader.init_embeddings(dim=dim, pretrained=embeddings_option)

    experiment["trainer_params"] = settings["trainer_params"]
    loss_settings = copy(settings["loss_params"])
    loss_settings["name"] = settings["loss-function"]
    experiment["loss_settings"] = settings["loss_params"]

    available_loss_funcs = {
        "BCE": BCEWithLogitsLoss,
        "RampNNPU": RampLossNNPU,
        "NNPU": NNPULoss
    }

    for iteration in range(settings["total-iterations"]):
        # This is the neural network
        # which will be used to classify the examples'
        estimator = CnnEstimator(
            pretrained_embeddings=pretrained
        )
        # estimator = MLP5(
        #     pretrained_embeddings=pretrained,
        #     max_document_len=datamodule.vectorizer.max_len
        # )

        # Create wrappers for nnPU loss & estimator
        params = settings["loss_params"]

        try:
            loss_fn_name = settings["loss-function"]
        except KeyError as key_err:
            print(error(key_err))

        if loss_fn_name == "BCE":
            loss_fn = BCEWithLogitsLoss()
        else:
            loss_fn = available_loss_funcs[loss_fn_name](**params)
        
        pu_net = PUNet(
            estimator=estimator,
            learning_rate=settings["learning_rate"],
            loss_fn=loss_fn
        )

        # Trainer is used
        # to specify model checkpoints,
        # training epochs, etc ...
        params = settings["trainer_params"]
        trainer = pl.Trainer(**params)

        trainer.fit(pu_net, datamodule)
        test_loss = trainer.test(datamodule=datamodule, ckpt_path="best")

        logits = trainer.predict(pu_net, datamodule.test_dataloader())
        logits = torch.concat(logits)
        probs, _ = torch.max(logits, 1)
        predictions = [1 if prob > 0.5 else 0 for prob in probs]

        clf_report = classification_report(datamodule.predict_labels, predictions, labels=[0, 1], output_dict=True)
        
        pu_labels = datamodule.cached_set["pu-label"].value_counts().to_dict()
        print(success(f"Iteration: {iteration} - Current labels: \t{pu_labels}"))
        clf_report["Current sample selection stats"] = pu_labels

        experiment[f"iteration-{iteration}"] = [test_loss[0], clf_report]

        # Break loop if positive 
        # augmentation is not desired
        if settings["biased_PU_assumption"]:
            break

        lgbm_params = settings["lgbm_params"]
        lgbm = LGBMClassifier(**lgbm_params)

        # Using the sample selector above
        # we extract positives to enrich the positive set
        if settings["total-iterations"] - 1 != iteration:
            purity, correct_flips, flipped = correct_label_issues(datamodule.cached_set, lgbm, n_jobs=1, folds=5)
            experiment["Sample purity"] = purity
            experiment["Correct flips"] = correct_flips
            experiment["Total flips"] = flipped

    
    save_logs(experiment)