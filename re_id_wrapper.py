import numpy as np
import random
import pandas as pd
import cv2 as cv
import time

# from centroids_reid.inference.create_embeddings import embed_from_directory

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
from copy import deepcopy

sys.path.append(".")

from centroids_reid.config import cfg

from centroids_reid.utils.reid_metric import get_dist_func

from centroids_reid.inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
)

from centroids_reid.config import cfg
from centroids_reid.train_ctl_model import CTLModel

from centroids_reid.inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    run_inference,
)


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

        with open("centroids_reid/create_embeddings_config.yml", "r") as stream:
            self.embed_args = yaml.safe_load(stream)

        if self.embed_args["config_file"] != "":
            cfg.merge_from_file(self.embed_args["config_file"])
        cfg.merge_from_list(self.embed_args["opts"])

        ### Build model
        self.model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)

        with open("centroids_reid/get_similar_config.yml", "r") as stream:
            self.similar_args = yaml.safe_load(stream)

        if self.similar_args["config_file"] != "":
            cfg.merge_from_file(self.similar_args["config_file"])
        cfg.merge_from_list(self.similar_args["opts"])

        self.connecting_dict = {}

        self.gallery_global_ids_list = []
        self.gallery_image_paths_list = []

        self.query_global_ids_list = []
        self.new_query_global_ids_list = []
        self.query_image_paths_list = []

        self.number_of_indexes = 0

        self.matched = None

        self.non_matched_ids = []
        self.non_matched_paths = []
        self.non_matched_embeddings = []

    def reservoir_sampling(self, result, iterator, k, n):
        """
        result: current samples chosen
        iterator: the stream list
        k: number of samples allowed
        n: current state of number of appends attempted
        """
        replace = False
        position = 0
        for item in iterator:
            if len(result) < k:
                result.append(item)
            else:
                # variable
                # j = int(random.random() * n)
                j = random.randrange(n)

                # if variable smaller than the allowed amount of elements
                if j < k:
                    # replace that exact element in the list with the item
                    replace = True
                    position = j
                    # result[j] = item

        return replace, position

    def embed_from_directory(self, gallery):

        ### Prepare logging
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        log = logging.getLogger(__name__)

        use_cuda = False  # if torch.cuda.is_available() and cfg.GPU_IDS else False

        if gallery == True:

            ### Inference
            log.info("Running inference")
            embeddings, paths = run_inference(
                self.model,
                cfg,
                print_freq=self.embed_args["print_freq"],
                use_cuda=use_cuda,
                list_of_paths=self.gallery_image_paths_list,
            )

            # ### Save
            # SAVE_DIR = Path(cfg.OUTPUT_DIR)
            # SAVE_DIR.mkdir(exist_ok=True, parents=True)

            # log.info(f"Saving embeddings and index to {str(SAVE_DIR)}")
            # np.save(SAVE_DIR / "embeddings.npy", embeddings)
            # # np.save(SAVE_DIR / "paths.npy", paths)

            self.gallery_embeddings = embeddings

        if gallery == False:

            # print("HOCUS POICUS")

            ### Inference
            log.info("Running inference")
            embeddings, paths = run_inference(
                self.model,
                cfg,
                print_freq=self.embed_args["print_freq"],
                use_cuda=use_cuda,
                list_of_paths=self.query_image_paths_list,
            )

            # SAVE_DIR = Path(cfg.OUTPUT_DIR)
            # SAVE_DIR.mkdir(exist_ok=True, parents=True)

            # log.info(f"Saving results to {str(SAVE_DIR)}")
            # np.save(SAVE_DIR / "query_embeddings.npy", embeddings)
            # np.save(SAVE_DIR / "query_paths.npy", paths)
            # print("HOCUS POICU2S")

            self.query_embeddings = embeddings

    def initialise(self, boxes, frames):

        """
        Function initialises the gallery infomation
        """

        # self.connecting_df = pd.DataFrame(
        #     columns=["global_id", "index", "1", "2", "3", "4", "5", "n"]
        # )

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

                if (
                    significant.loc[detection, "cam_id"] == 2
                    and significant.loc[detection, "local_id"] == 13
                ):
                    print("\n\n THIS IS THE PATH FOR 13:", path)

                if (
                    significant.loc[detection, "cam_id"] == 2
                    and significant.loc[detection, "local_id"] == 86
                ):
                    print("\n\n THIS IS THE PATH FOR 86:", path)

                try:
                    indexes_of_id = self.connecting_dict[
                        significant.loc[detection, "global_id"]
                    ]["indices"]
                    if 0 < len(indexes_of_id) < 5:
                        # add the index
                        indexes_of_id.append(self.number_of_indexes)
                        self.connecting_dict[significant.loc[detection, "global_id"]][
                            "image_paths"
                        ].append(path)
                        self.connecting_dict[significant.loc[detection, "global_id"]][
                            "n"
                        ] += 1

                    else:
                        NotImplementedError

                except:
                    self.connecting_dict[significant.loc[detection, "global_id"]] = {
                        "indices": [self.number_of_indexes],
                        "image_paths": [path],
                        "n": 1,
                    }

                cv.imwrite(
                    path,
                    detection_im,
                )

                self.number_of_indexes += 1

        self.embed_from_directory(gallery=True)

    def save_images(self, boxes, frames):

        """
        Save images to become a query set everntually. ALL images are saved by concatenating time on.
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

                if (
                    significant.loc[detection, "cam_id"] == 2
                    and significant.loc[detection, "local_id"] == 13
                ):
                    print("\n\n THIS IS THE PATH FOR 13:", path)

                if (
                    significant.loc[detection, "cam_id"] == 2
                    and significant.loc[detection, "local_id"] == 86
                ):
                    print("\n\n THIS IS THE PATH FOR 86:", path)

                self.query_image_paths_list.append(path)

                cv.imwrite(
                    path,
                    detection_im,
                )

    def find_similar_people(self):
        ### Prepare logging

        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        log = logging.getLogger(__name__)

        # ### Data preparation
        # if args["images_in_subfolders"]:
        #     dataset_type = ImageFolderWithPaths
        # else:
        #     dataset_type = ImageDataset
        # log.info(f"Preparing data using {type(dataset_type)} dataset class")
        # val_loader = make_inference_data_loader(cfg, cfg.DATASETS.ROOT_DIR, dataset_type)

        ### Load gallery data
        # LOAD_PATH = Path(args["gallery_data"])
        embeddings_gallery = torch.from_numpy(self.gallery_embeddings)
        # torch.from_numpy(
        #     np.load(LOAD_PATH / "embeddings.npy", allow_pickle=True)
        # )
        paths_gallery = np.array(
            self.gallery_image_paths_list
        )  # np.load(LOAD_PATH / "paths.npy", allow_pickle=True)

        paths = np.array(
            self.query_image_paths_list
        )  # np.load(LOAD_PATH / "query_paths.npy", allow_pickle=True)

        embeddings = torch.from_numpy(self.query_embeddings)
        # torch.from_numpy(
        #     np.load(LOAD_PATH / "query_embeddings.npy", allow_pickle=True)
        # )

        if self.similar_args["normalize_features"]:
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

        print("gallery shape", embeddings_gallery.shape)
        print("embeddings shape", embeddings.shape)

        ### Calculate similarity
        t1 = time.time()

        log.info("Calculating distance and getting the most similar ids per query")
        dist_func = get_dist_func(cfg.SOLVER.DISTANCE_FUNC)
        distmat = dist_func(x=embeddings, y=embeddings_gallery).cpu().numpy()
        indices = np.argsort(distmat, axis=1)

        print("time to complete search", time.time() - t1)

        ### Constrain the results to only topk most similar ids
        indices = (
            indices[:, : self.similar_args["topk"]]
            if self.similar_args["topk"]
            else indices
        )

        threshold = 0.2

        out = {}

        self.non_matched_ids = []
        self.non_matched_paths = []
        self.non_matched_embeddings = []

        self.new_query_global_ids_list = deepcopy(self.query_global_ids_list)

        for q_num, query_path in enumerate(paths):
            mask = distmat[q_num, indices[q_num, :]] < threshold

            if np.all(mask == False):
                self.non_matched_ids.append(self.query_global_ids_list[q_num])
                self.non_matched_paths.append(query_path)
                self.non_matched_embeddings.append(embeddings[q_num, :].cpu().numpy())

            else:
                query_dict = {}
                query_dict["indices"] = indices[q_num, :][mask]
                # print(indices)
                # print(indices[q_num, :][mask])
                query_dict["paths"] = paths_gallery[indices[q_num, :]][mask]
                query_dict["distances"] = distmat[q_num, indices[q_num, :]][mask]

                query_dict["matched_global_ids"] = np.array(
                    self.gallery_global_ids_list
                )[indices[q_num, :][mask]]
                query_dict["embedding"] = embeddings[q_num, :].cpu().numpy()
                query_dict["query_global_id"] = self.query_global_ids_list[q_num]
                out[query_path] = query_dict

                # this list contains all the new ids associated with the original query ids
                self.new_query_global_ids_list[q_num] = np.array(
                    self.gallery_global_ids_list
                )[indices[q_num, :][mask]][0]

        self.matched = out

        print(out)
        ### Save
        SAVE_DIR = Path(cfg.OUTPUT_DIR)
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        log.info(f"Saving results to {str(SAVE_DIR)}")
        np.save(SAVE_DIR / "results.npy", out)
        np.save(SAVE_DIR / "query_paths.npy", paths)

        self.match_global_ids()

        return self.query_global_ids_list, self.new_query_global_ids_list

    def match_global_ids(self):
        """
        Process the results from the get similar function.
        """
        # two things need to happen 1) The index needs to be added (or resivour sampled) for an addition to the gallery data. Also, we then need to append the detection to the gallery_embeddings, the gallery_global_ids and the gallery image paths.

        # Also, we need to go find all the query images that did NOT have a match meeting the threshold and then subsequently add them to the gallery_embeddings, the gallery_global_ids and the gallery_image paths with their original global_ids which were either supplied by the tracker (ocsort) or by the nearest neighbour in terms of location.

        list_length = 5
        # print(self.matched)
        # exit()

        # these are for all the nice matches.
        for key in list(self.matched.keys()):

            query_image_path = key

            # query_image_path = self.matched[key]["paths"]
            master_global_id = self.matched[key]["matched_global_ids"][0]

            # if the gloabl_id already exists, but is not full yet:
            if len(self.connecting_dict[master_global_id]["indices"]) < list_length:

                # we immediately add the embeddings, the global_ids, and the gallery_image_paths

                # matched global_id
                self.gallery_global_ids_list.append(master_global_id)
                # the embedding
                self.gallery_embeddings = np.concatenate(
                    (
                        self.gallery_embeddings,
                        np.expand_dims(self.matched[key]["embedding"], axis=0),
                    ),
                    axis=0,
                )
                # the query path
                self.gallery_image_paths_list.append(query_image_path)

                # the index at which to insert these
                # append the length (the last entry position)
                self.connecting_dict[master_global_id]["indices"].append(
                    len(self.gallery_global_ids_list) - 1
                )
                self.connecting_dict[master_global_id]["image_paths"].append(
                    query_image_path
                )
                # add the number of elements added to the connecting dict
                self.connecting_dict[master_global_id]["n"] += 1

            if len(self.connecting_dict[master_global_id]["indices"]) == list_length:
                # now do some fancy resivour sampling to determine if we want to include the sample
                # The entry does not actually matter, and the list just has one entry
                iterator = [0]
                # print(self.connecting_dict[master_global_id]["indices"])
                replace_boolean, position = self.reservoir_sampling(
                    self.connecting_dict[master_global_id]["indices"],
                    iterator,
                    list_length,
                    self.connecting_dict[master_global_id]["n"],
                )
                # print(replace_boolean, position)

                if replace_boolean == True:

                    # print("yes entered")
                    # the resivour sampling is successful, so we need to replace the embedding as well as the image path:)
                    index = self.connecting_dict[master_global_id]["indices"][position]

                    self.gallery_embeddings[index, :] = self.matched[key]["embedding"]
                    # the query path
                    self.connecting_dict[master_global_id]["image_paths"][
                        position
                    ] = query_image_path

                    self.gallery_image_paths_list[index] = query_image_path

                # discard the embedding, even though it might have been cool
                else:
                    pass

                # add the number of elements added to the connecting dict
                self.connecting_dict[master_global_id]["n"] += 1

        # for all the query images that had no match whatsoever, we need to add them to the embeddings, the image_paths, the global_id and finally the connecting df
        for idx, id in enumerate(self.non_matched_ids):
            # add the ids
            self.gallery_global_ids_list.append(id)

            self.gallery_embeddings = np.concatenate(
                (
                    self.gallery_embeddings,
                    np.expand_dims(self.non_matched_embeddings[idx], axis=0),
                ),
                axis=0,
            )
            self.gallery_image_paths_list.append(self.non_matched_paths[idx])

            # now also add the new entries to the connecting dictionary.

            self.connecting_dict[id] = {
                "indices": [len(self.gallery_global_ids_list) - 1],
                "image_paths": [self.non_matched_paths[idx]],
                "n": 1,
            }

        print(self.connecting_dict)

        return 0


def reid(
    reid_gallery,
    arg_queue,
    que,
):
    """
    Applies re-identification (ReID) to track people across frames.

    Args:
        reid_gallery (ReIDGallery): An object representing the ReID gallery.
        view_id_table (pd.DataFrame): A Pandas DataFrame mapping person IDs to global IDs across views.
        frame_id (int): The index of the current frame in the video sequence.
        frames (List[np.ndarray]): A list of frames in the video sequence.
        boxes_with_probabilities A list of bounding boxes (in the format (x1, y1, x2, y2)) and their associated probabilities for the current frame.
        context_dataframe (pd.DataFrame): A Pandas DataFrame

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] or None: If the ReID process is complete, returns a tuple containing the updated context dataframe and view ID table. Otherwise, returns None.
    """

    # intitalialise the gallery object
    while True:
        if not arg_queue.empty():
            (
                frame_id,
                frames,
                boxes_with_probabilities,
            ) = arg_queue.get()

            print("passing frame::::", frame_id)

            if frame_id == 0:
                # initialise
                reid_gallery.initialise(boxes_with_probabilities, frames)
                print("done with 1")

                # return (context_dataframe, view_id_table)

                # que.put(context_dataframe)
                # que.put(view_id_table)
                # que.put(original_id_list)
                # que.put(new_query_ids)

                # que.put(reid_gallery)

                # return reid_gallery
                que.put("comp")

            if frame_id % 5 == 0 and frame_id > 0 and frame_id % 45 != 0:
                print("starting with 2")

                reid_gallery.save_images(boxes_with_probabilities, frames)
                print("done with 2")

                #     # save all new bboxes
                #     self.save_images(boxes_with_probabilities, frames)

                #     # print("HELLOOOOOOOO")
                #     # if frame_id % 50 != 0:
                #     # return context_dataframe, view_id_table
                #     # que.put(context_dataframe)
                #     # que.put(view_id_table)

                #     # que.put(original_id_list)
                #     # que.put(new_query_ids)
                #     # return reid_gallery
                que.put("comp")

            if (frame_id % 50 == 0) and (frame_id > 0):

                print("ENTEREDDDD the finder")
                reid_gallery.embed_from_directory(gallery=False)

                # print("gallery embeddigs:::::::")
                # print(reid_gallery.gallery_embeddings)

                # print(reid_gallery.query_image_paths_list)

                # embed the images that were saved into the latent space
                # reid_gallery.embed_from_directory(gallery=False)

                # run the similarity thing
                original_id_list, new_query_ids = reid_gallery.find_similar_people()

                # update the context df things
                # context_dataframe["global_id"] = context_dataframe["global_id"].replace(
                #     original_id_list, new_query_ids
                # )

                # view_id_table["global_id"] = view_id_table["global_id"].replace(
                #     original_id_list, new_query_ids
                # )

                reid_gallery.query_image_paths_list = []
                reid_gallery.query_global_ids_list = []

                # print(original_id_list)
                # print(new_query_ids)

                if original_id_list == new_query_ids:
                    print("\nidentical")

                else:
                    print("\n replaced!!")

                # return context_dataframe, view_id_table
                # que.put(reid_gallery)
                que.put(original_id_list)
                que.put(new_query_ids)
            else:
                que.put("comp")
