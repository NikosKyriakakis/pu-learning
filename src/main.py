from lightgbm import LGBMClassifier

from configuration import *
from label_noise.clean_labels_helpers import *
from pulearn.neural_nets.data_modules import TextDataModule
from pulearn.neural_nets.estimators import *
from src.pulearn.neural_nets.loss_functions.nnpu_loss import NNPULoss
from textprep.embed import EmbeddingLoader


def save_logs(output):
    with open("logs.txt", "w") as checkpoint:
        json.dump(output, checkpoint, indent=4)


# Driver code
if __name__ == "__main__":
    # Load configuration settings
    settings = load_settings("app-settings.json")

    # Create a download manager that handles
    # downloading the pretrained embeddings, datasets, etc ...
    download_mgr = DownloadManager(settings)

    # The text data module is the main
    # component that prepares and provides
    # data to the deep learning model which will be used
    datamodule = TextDataModule(
        download_mgr=download_mgr,
        csv_file="PU_20News.csv",
        input_field="text",
        target_field="label",
        negative_value=-1,
        dev_run=False,
        dataloader_params={
            "batch_size": 128,
            "num_workers": int(os.cpu_count() / 2)
        }
    )

    # The embedding loader currently supports
    # loading GloVe & FastText pretrained embeddings
    loader = EmbeddingLoader(
        datamodule.vectorizer.text_vocab,
        mgr=download_mgr
    )

    pretrained_embedding_options = settings["pretrained_embedding"]
    pretrained = loader.init_embeddings(
        pretrained=pretrained_embedding_options["option"],
        dim=pretrained_embedding_options["dim"]
    )

    logs = {}
    for iteration in range(7):
        # This is the neural network 
        # which will be used to classify the examples 
        estimator = CnnEstimator(pretrained_embeddings=pretrained)

        # sample_selector = RandomForestClassifier(n_jobs=-1)
        sample_selector = LGBMClassifier(n_jobs=-1)

        # Using the sample selector above
        # we extract positives to enrich the positive set
        correct_label_issues(datamodule, sample_selector, n_jobs=5)

        # Create wrappers for nnPU loss & estimator
        loss_fn = NNPULoss(prior=0.44, gamma=1, beta=0, positive_class=1)
        pu_net = PUNet(
            estimator=estimator,
            learning_rate=0.001,
            loss_fn=loss_fn
        )

        # Trainer is used
        # to specify model checkpoints,
        # training epochs, etc ...
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=1,
            precision=16,
            # fast_dev_run=True
        )

        trainer.fit(pu_net, datamodule)
        results = trainer.test(datamodule=datamodule, ckpt_path="best")

        pu_labels = datamodule.documents["pu-label"].value_counts()
        print(success(f"Iteration: {iteration} - Current labels: \t{pu_labels.to_dict()}"))
        logs[iteration] = {"Sample selection stats": pu_labels.to_dict(), "Test Results": results}

    save_logs(logs)
