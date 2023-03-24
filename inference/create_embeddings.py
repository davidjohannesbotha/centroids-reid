import argparse
import logging
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import yaml

sys.path.append(".")

from centroids_reid.config import cfg
from centroids_reid.train_ctl_model import CTLModel

from centroids_reid.inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    calculate_centroids,
    create_pid_path_index,
    run_inference,
)


def embed_from_directory(list_of_paths, gallery):

    ### Prepare logging
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    log = logging.getLogger(__name__)

    with open("centroids_reid/create_embeddings_config.yml", "r") as stream:
        args = yaml.safe_load(stream)

    if args["config_file"] != "":
        cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])

    ### Build model
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)

    use_cuda = True  # if torch.cuda.is_available() and cfg.GPU_IDS else False

    ### Inference
    log.info("Running inference")
    embeddings, paths = run_inference(
        model,
        cfg,
        print_freq=args["print_freq"],
        use_cuda=use_cuda,
        list_of_paths=list_of_paths,
    )

    ### Create centroids
    # log.info("Creating centroids")
    # if cfg.MODEL.USE_CENTROIDS:
    #     pid_path_index = create_pid_path_index(paths=paths, func=exctract_func)
    #     embeddings, paths = calculate_centroids(embeddings, pid_path_index)

    if gallery == True:
        ### Save
        SAVE_DIR = Path(cfg.OUTPUT_DIR)
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        log.info(f"Saving embeddings and index to {str(SAVE_DIR)}")
        np.save(SAVE_DIR / "embeddings.npy", embeddings)
        np.save(SAVE_DIR / "paths.npy", paths)

    if gallery == False:

        SAVE_DIR = Path(cfg.OUTPUT_DIR)
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        log.info(f"Saving results to {str(SAVE_DIR)}")
        np.save(SAVE_DIR / "query_embeddings.npy", embeddings)
        np.save(SAVE_DIR / "query_paths.npy", paths)
