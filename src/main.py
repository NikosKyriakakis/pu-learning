from pulearn.neural_nets.data_modules import TextDataModule
from textprep.embed import EmbeddingLoader
from pulearn.neural_nets.estimators import *
from configuration import *
from cleanlab.classification import *
from label_noise.clean_labels_helpers import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import pytorch_lightning as pl


# Driver code
if __name__ == "__main__":
    # Load configuration settings
    settings = load_settings("appsettings.json")

    # Create a download manager that handles
    # downloading the pretrained embeddings, datasets, etc ...
    download_mgr = DownloadManager(settings)

    # The text data module is the main
    # component that prepares and provides
    # data to the deep learning model which will be used
    datamodule = TextDataModule (
        download_mgr=download_mgr, 
        csv_file="imdb.txt", 
        input_field="Text", 
        target_field="Sentiment",
        negative_value=0,
        dev_run=True,
        dataloader_params={
            "batch_size": 64, 
            "num_workers": int(os.cpu_count() / 2)
        }
    )

    # The embedding loader currently supports
    # loading GloVe & FastText pretrained embeddings
    loader = EmbeddingLoader (
        datamodule.vectorizer.text_vocab,
        mgr=download_mgr
    )

    pretrained_embedding_options = settings["pretrained_embedding"]
    pretrained = loader.init_embeddings (
        pretrained=pretrained_embedding_options["option"], 
        dim=pretrained_embedding_options["dim"]
    )

    # This is the neural network 
    # which will be used to classify the examples 
    estimator = CNNEstimator(num_classes=1, pretrained_embedding=pretrained)

    pu_labels = datamodule.documents["pu-label"].value_counts()
    print(success(f"Labels before applying cleanlab: \t{pu_labels.to_dict()}"))

    rf = RandomForestClassifier(n_jobs=-1)
    correct_label_issues(datamodule, rf)

    pu_labels = datamodule.documents["pu-label"].value_counts()
    print(success(f"Labels after applying cleanlab: \t{pu_labels.to_dict()}"))

    # # This is an nnPU wrapper 
    # # object used to specify 
    # # nnPU hyperparameters 
    # # such as β, γ, prior, ...
    # pu_net = NNPUNet (
    #     estimator=estimator, 
    #     learning_rate=0.001,
    #     prior=0.44
    # )

    # # Trainer is used
    # # to specify model checkpoints,
    # # training epochs, etc ...
    # trainer = pl.Trainer (
    #     max_epochs=10,
    #     # fast_dev_run=True
    # )

    # trainer.fit(pu_net, datamodule)

    # trainer.test(datamodule=datamodule)