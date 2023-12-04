import importlib
import json
import os
import random
from typing import *

import matplotlib.pyplot as plt
import numpy.random
import torch
from data import (
    ATCCompleteData,
    ATCO2SimData,
    ATCOSimData,
    ZCUCZATCDataset,
    Data,
    get_train_test_split,
)
from models import Model

# root dataset paths corresponding to data analysis classes
datasets: Dict[str, Data] = {
    # TODO: find a way to sync file paths across computers (shell/env var, config file?)
    "/home/vandebra/programming/thesis_data/atc0_comp": ATCCompleteData,
    "/home/vandebra/programming/thesis_data/atcosim/": ATCOSimData,
    "/home/vandebra/programming/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
    "/home/vandebra/programming/thesis_data/ZCU_CZ_ATC": ZCUCZATCDataset,
}

nsp_datasets: Dict[str, Data] = {
    "/home/vandebra/programming/thesis_data/atc0_comp": ATCCompleteData,
}


def parse_datasets():
    # collection of dataset stats. using a dictionary so it's easier to dump to JSON or YAML later
    dataset_info = {"dataset_info": [], "trimmed_dataset_info": []}
    data_objects: List[Data] = []

    # initializes each implementing class with its data which is specified by `root_path`
    # see `datasets` and implementing classes for more details
    for root_path, data_class in datasets.items():
        # initialize class with root path and random seed
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)

        # parse dataset info, along with some printout so the user has some idea of what's
        # happening
        print(f"Parsing transcripts for '{data_analysis.name}'", end="...")
        data_analysis.parse_transcripts()
        print("Done")

        data_analysis.summary()
        # dump corpora to a standard corpus.txt style file, we'll have one for each dataset and
        # one for everything combined together
        corpus_filename = f"{data_analysis.name.replace(' ', '_')}.txt"
        print(f"Dumping dataset to corpus: {corpus_filename}", end="...")
        data_analysis.dump_corpus(f"corpora/{corpus_filename}")
        print("Done")

        dataset_info["dataset_info"].append(data_analysis.dump_info())
        # data_analysis.plot_histogram()

        dataset_info["trimmed_dataset_info"].append(data_analysis.dump_info())

        data_objects.append(data_analysis)

    # for root_path, data_class in nsp_datasets.items():
    #     nsp_dataset: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)

    #     print(f"Parsing transcripts for NSP tasks for '{nsp_dataset.name}'")
    #     nsp_dataset.parse_transcripts_for_nsp()
    #     print("Done")

    # nsp_dataset.summary()

    # write stats to a json file
    os.makedirs("corpora", exist_ok=True)
    with open("corpora/dataset_stats.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_info, indent=1))

    # concatenate everything to first object
    for o in data_objects[1:]:
        data_objects[0].concat(o)

    data_objects[0].dataset_name = "All"
    data_objects[0].summary()

    outliers = data_objects[0].detect_outliers()
    print(f"Mean outlier sequence length: {outliers['seq_len'].mean()}")
    print(f"Min outlier sequence length: {outliers['seq_len'].min()}")
    print(f"Max outlier sequence length: {outliers['seq_len'].max()}")
    print(f"Std outlier sequence length: {outliers['seq_len'].std()}")
    print(
        f"Most significant outlier:\n{outliers.where(outliers['seq_len'] == outliers['seq_len'].max()).dropna()}"
    )

    print(f"Max LOF score: {outliers['negative_outlier_factor_'].min()}")
    print(
        outliers["text"]
        .where(
            outliers["negative_outlier_factor_"]
            == outliers["negative_outlier_factor_"].min()
        )
        .dropna()
    )

    data_objects[0].summary()

    with open("outliers.txt", "w") as f:
        for item, length in zip(
            outliers["text"].tolist(), outliers["seq_len"].tolist()
        ):
            f.write(f"{int(length)}\n")
            f.write(item + "\n")

    # data_objects[0].plot_histogram()

    # split data into train and test
    print("Generating train/test split", end="...")
    train, valid, test = get_train_test_split(
        data_objects[0], shuffle=True, random_seed=RANDOM_SEED
    )
    print("Done")

    # dump corpra to files
    print(f"Dumping concatenated data to all_corpus.txt", end="...")
    data_objects[0].dump_corpus("corpora/all_corpus.txt")
    print("Done")

    print("Dumping training corpus to train_corpus.txt", end="...")
    train.dump_corpus("corpora/train_corpus.txt")
    print("Done")

    print("Dumping validation corpus to validation_corpus.txt", end="...")
    valid.dump_corpus("corpora/validation_corpus.txt")
    print("Done")

    print("Dumping testing corpus to test_corpus.txt", end="...")
    test.dump_corpus("corpora/test_corpus.txt")
    print("Done")

    return train, valid, test


if __name__ == "__main__":
    plt.style.use("ggplot")
    RANDOM_SEED: int = 1

    # control sources of randomness
    numpy.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train, valid, test = parse_datasets()

    # python representation of pretraining_info.json
    # pretraining_info = None

    # if os.path.exists("corpora/all_corpus.txt"):
    #     data = Data.from_corpus("corpora/all_corpus.txt")
    #     train, valid, test = get_train_test_split(data)
    # else:
    #     train, valid, test = parse_datasets(dataset_info)

    # # load entries from pretraining_info.json
    # with open("config/pretraining_info.json", "r", encoding="utf-8") as f:
    #     pretraining_info = json.load(f)

    # test_results = {"test_results": []}

    # with open("config/testing_config.json", "r", encoding="utf-8") as f:
    #     testing_info = json.load(f)

    # for model_config in testing_info["testing_config"]:
    #     module_name = model_config["model_class"].rsplit(".")
    #     module = __import__(".".join(module_name[:-1]), fromlist=[module_name[-1]])
    #     model_class: Model = getattr(module, module_name[-1])

    #     model = model_class.load_from(
    #         path=model_config["checkpoint_folder_path"],
    #         train_dataset=train,
    #         valid_dataset=valid,
    #         checkpoint_name=model_config["checkpoint_name"],
    #     )

    #     results = model.test(test)

    #     test_results["test_results"].append(
    #         {
    #             "checkpoint_name": model_config["checkpoint_name"],
    #             "model_class": model_config["model_class"],
    #             "results": results,
    #         }
    #     )

    # with open("test_results.json", "w", encoding="utf-8") as f:
    #     json.dump(test_results, f)

    # for model_config in pretraining_info["pretraining_config"]:
    #     """dynamically import class specified in "model_class"""
    #     # split class from module since the mechanics of importing the two differ
    #     module_name = model_config["model_class"].rsplit(".")
    #     # import module name and specify the class in the from list e.g. from module_name import class
    #     module = __import__(".".join(module_name[:-1]), fromlist=[module_name[-1]])
    #     # get the class from the imported module
    #     model_class: Model = getattr(module, module_name[1])

    #     if os.path.exists(model_config["checkpoint_folder_path"]):
    #         model = model_class.load_from(
    #             path=model_config["checkpoint_folder_path"],
    #             train_dataset=train,
    #             valid_dataset=valid,
    #             checkpoint_name=model_config["checkpoint_name"],
    #         )
    #     else:
    #         model = model_class(train_dataset=train, valid_dataset=valid)

    #     model.fit(max_epochs=model_config["max_epochs"])
    #     model.save_to(model_config["checkpoint_folder_path"])
