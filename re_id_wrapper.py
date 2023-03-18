import numpy as np
import random
import pandas as pd
import cv2 as cv
import time

from centroids_reid.inference.create_embeddings import embed_from_directory
from centroids_reid.inference.get_similar import find_similar_people

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch

sys.path.append(".")

from centroids_reid.config import cfg
from centroids_reid.train_ctl_model import CTLModel
from centroids_reid.utils.reid_metric import get_dist_func

from centroids_reid.inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    make_inference_data_loader,
    run_inference,
)
import time


def reservoir_sampling(iterator, k):
    result = []
    n = 0
    for item in iterator:
        n = n + 1
        # k = allowed number
        if len(result) < k:
            result.append(item)
        else:
            # variable
            j = int(random.random() * n)
            # if variable smaller than the allowed amount of elements
            if j < k:
                # replace that exact element in the list with the item
                result[j] = item

    return result


stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

k = 3
print(reservoir_sampling(stream, k))


class gallery:
    def __init__(self, config):
        """
        Class that encapsulates all the data stream related things.
        """
        self.image_gallery_path = config["image_gallery"]
        self.image_query_gallery_path = config["image_query_gallery"]

        self.embedding_gallery_path = config["embedding_gallery"]
        self.embedding_paths = config["embedding_paths"]

        self.query_embeddings_path = config["query_embeddings"]
        self.query_paths = config["query_paths"]

        # gallery embeddings to stroe in memeory
        self.gallery_embeddings = None
        # query embeddings to be stored in memory
        self.query_embeddings = None

        # the dataframe that will connect everything
        self.connecting_df = pd.DataFrame(
            columns=["global_id", "index", "1", "2", "3", "4", "5", "n"]
        )
        self.gallery_global_ids_list = []
        self.gallery_image_paths_list = []

        self.query_global_ids_list = []
        self.query_image_paths_list = []

    def initialise(self, boxes, frames):

        """
        Function initialises the gallery infomation
        """

        for cam in range(len(boxes)):
            significant = boxes[cam][
                ["global_id", "local_id", "cam_id", "x_1", "x_2", "y_1", "y_2"]
            ]

            # print(significant)
            for detection in range(len(significant.global_id)):

                detection_im = frames[cam][
                    significant.loc[detection, "y_1"]
                    .astype(int) : (
                        significant.loc[detection, "y_1"]
                        + significant.loc[detection, "y_2"]
                    )
                    .astype(int),
                    significant.loc[detection, "x_1"]
                    .astype(int) : (
                        significant.loc[detection, "x_1"]
                        + significant.loc[detection, "x_2"]
                    )
                    .astype(int),
                ]

                self.gallery_global_ids_list.append(
                    significant.loc[detection, "global_id"]
                )

                path = (
                    self.image_gallery_path
                    + "/"
                    + str(time.time())
                    + "_"
                    + str(int(significant.loc[detection, "cam_id"]))
                    + str(int(significant.loc[detection, "local_id"]))
                    + ".jpg"
                )
                self.gallery_image_paths_list.append(path)

                cv.imwrite(
                    path,
                    detection_im,
                )

        embed_from_directory(self.gallery_image_paths_list, gallery=True)

    def create_queries(self, boxes, frames):

        """
        Function initialises the gallery infomation
        """
        q = 0
        for cam in range(len(boxes)):
            significant = boxes[cam][
                ["global_id", "local_id", "cam_id", "x_1", "x_2", "y_1", "y_2"]
            ]

            # print(significant)
            for detection in range(len(significant.global_id)):

                detection_im = frames[cam][
                    significant.loc[detection, "y_1"]
                    .astype(int) : (
                        significant.loc[detection, "y_1"]
                        + significant.loc[detection, "y_2"]
                    )
                    .astype(int),
                    significant.loc[detection, "x_1"]
                    .astype(int) : (
                        significant.loc[detection, "x_1"]
                        + significant.loc[detection, "x_2"]
                    )
                    .astype(int),
                ]

                self.query_global_ids_list.append(
                    significant.loc[detection, "global_id"]
                )

                path = (
                    self.image_query_gallery_path
                    + "/"
                    + str(time.time())
                    + "_"
                    + str(int(significant.loc[detection, "cam_id"]))
                    + str(int(significant.loc[detection, "local_id"]))
                    + ".jpg"
                )
                self.query_image_paths_list.append([q, path])

                cv.imwrite(
                    path,
                    detection_im,
                )
                q += 1

        embed_from_directory(self.gallery_image_paths_list, gallery=False)

    def find_similar_people(self):
        ### Prepare logging

        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        log = logging.getLogger(__name__)

        with open("centroids_reid/get_similar_config.yml", "r") as stream:
            args = yaml.safe_load(stream)

        if args["config_file"] != "":
            cfg.merge_from_file(args["config_file"])
        cfg.merge_from_list(args["opts"])

        ### Data preparation
        if args["images_in_subfolders"]:
            dataset_type = ImageFolderWithPaths
        else:
            dataset_type = ImageDataset
        log.info(f"Preparing data using {type(dataset_type)} dataset class")
        # val_loader = make_inference_data_loader(cfg, cfg.DATASETS.ROOT_DIR, dataset_type)

        ### Load gallery data
        LOAD_PATH = Path(args["gallery_data"])
        embeddings_gallery = torch.from_numpy(
            np.load(LOAD_PATH / "embeddings.npy", allow_pickle=True)
        )
        paths_gallery = np.load(LOAD_PATH / "paths.npy", allow_pickle=True)

        paths = np.load(LOAD_PATH / "query_paths.npy", allow_pickle=True)

        # print(np.array(paths))
        # exit()

        embeddings = torch.from_numpy(
            np.load(LOAD_PATH / "query_embeddings.npy", allow_pickle=True)
        )

        if args["normalize_features"]:
            embeddings_gallery = torch.nn.functional.normalize(
                embeddings_gallery, dim=1, p=2
            )
            embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2)
        else:
            embeddings = torch.from_numpy(embeddings)

        # Use GPU if available
        # device = torch.device("cuda") if cfg.GPU_IDS else torch.device("cpu")
        device = torch.device("cpu")

        embeddings_gallery = embeddings_gallery.to(device)
        embeddings = embeddings.to(device)

        ### Calculate similarity
        t1 = time.time()

        log.info("Calculating distance and getting the most similar ids per query")
        dist_func = get_dist_func(cfg.SOLVER.DISTANCE_FUNC)
        distmat = dist_func(x=embeddings, y=embeddings_gallery).cpu().numpy()
        indices = np.argsort(distmat, axis=1)
        print("time to complete search", time.time() - t1)

        ### Constrain the results to only topk most similar ids
        indices = indices[:, : args["topk"]] if args["topk"] else indices

        out = {
            query_path: {
                "indices": indices[q_num, :],
                "paths": paths_gallery[indices[q_num, :]],
                "distances": distmat[q_num, indices[q_num, :]],
            }
            for q_num, query_path in enumerate(paths)
        }

        ### Save
        SAVE_DIR = Path(cfg.OUTPUT_DIR)
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        log.info(f"Saving results to {str(SAVE_DIR)}")
        np.save(SAVE_DIR / "results.npy", out)
        np.save(SAVE_DIR / "query_embeddings.npy", embeddings)
        np.save(SAVE_DIR / "query_paths.npy", self.image_query_gallery_path)
