from pulearn.neural_nets.data_modules import TextDataModule
from textprep.embed import EmbeddingLoader
from pulearn.neural_nets.estimators import *
from configuration import *
from cleanlab.classification import *
from label_noise.clean_labels_helpers import *

from sklearn.ensemble import RandomForestClassifier
# from skorch import NeuralNetClassifier
# from pulearn.neural_nets.loss_functions.adaptive_loss import AdaptiveLossFunctionMod

import pytorch_lightning as pl
import json


def save_logs(logs):
    with open("logs.txt", "w") as checkpoint:
        json.dump(logs, checkpoint, indent=4)


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
        dev_run=False,
        dataloader_params={
            "batch_size": 128, 
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

    logs = {}
    for iteration in range(3):
        # This is the neural network 
        # which will be used to classify the examples 
        estimator = CNNEstimator(num_classes=1, pretrained_embedding=pretrained)

        sample_selector = RandomForestClassifier(n_jobs=1)
        # adaptive_loss = AdaptiveLossFunctionMod(device="cuda:0", num_dims=3, float_dtype=np.float32)

        # sample_selector = NeuralNetClassifier (
        #     MLP5 (
        #         pretrained_embedding=pretrained, 
        #         max_len=datamodule.vectorizer.max_len
        #     ), 
        #     criterion=nn.CrossEntropyLoss, 
        #     device="cuda"
        # )

        # sample_selector = NeuralNetClassifier (
        #     CNNEstimator (
        #         num_classes=2, 
        #         pretrained_embedding=pretrained, 
        #         apply_softmax=True
        #     ), 
        #     criterion=nn.CrossEntropyLoss, 
        #     device="cuda"
        # )
        
        correct_label_issues(datamodule, sample_selector, n_jobs=5)

        # This is an nnPU wrapper 
        # object used to specify 
        # nnPU hyperparameters 
        # such as β, γ, prior, ...
        pu_net = NNPUNet (
            estimator=estimator, 
            learning_rate=0.001,
            prior=0.5
        )

        # Trainer is used
        # to specify model checkpoints,
        # training epochs, etc ...
        trainer = pl.Trainer (
            max_epochs=250,
            accelerator="gpu",
            devices=1,
            precision=16
            # fast_dev_run=True
        )

        trainer.fit(pu_net, datamodule)

        results = trainer.test(datamodule=datamodule, ckpt_path="best")

        pu_labels = datamodule.documents["pu-label"].value_counts()

        print(success(f"Current labels: \t{pu_labels.to_dict()}"))
    
        logs[iteration] = { "Sample selection stats": pu_labels.to_dict(),  "Test Results": results}
    
    save_logs(logs)