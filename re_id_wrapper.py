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
import multiprocessing as mp

from nw_utils.processing import import_global_ids


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

        self.threshold = self.similar_args["similarity_threshold"]

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

        self.unique_query_global_ids = []
        self.unique_gallery_global_ids = []

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

                # if (
                #     significant.loc[detection, "cam_id"] == 2
                #     and significant.loc[detection, "local_id"] == 13
                # ):
                #     print("\n\n THIS IS THE PATH FOR 13:", path)

                # if (
                #     significant.loc[detection, "cam_id"] == 2
                #     and significant.loc[detection, "local_id"] == 86
                # ):
                #     print("\n\n THIS IS THE PATH FOR 86:", path)

                self.query_image_paths_list.append(path)

                cv.imwrite(
                    path,
                    detection_im,
                )

    def find_similar_people(self):
        ### Prepare logging

        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        log = logging.getLogger(__name__)

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

        out = {}

        self.non_matched_ids = []
        self.non_matched_paths = []
        self.non_matched_embeddings = []

        self.unique_query_global_ids = []
        self.unique_gallery_global_ids = []

        self.new_query_global_ids_list = deepcopy(self.query_global_ids_list)

        for q_num, query_path in enumerate(paths):
            mask = distmat[q_num, indices[q_num, :]] < self.threshold

            if np.all(mask == False):
                self.non_matched_ids.append(self.query_global_ids_list[q_num])
                self.non_matched_paths.append(query_path)
                self.non_matched_embeddings.append(embeddings[q_num, :].cpu().numpy())

            else:
                query_dict = {}
                # query index
                query_dict["indices"] = indices[q_num, :][mask]
                # query path
                query_dict["paths"] = paths_gallery[indices[q_num, :]][mask]
                # distance from query to match
                query_dict["distances"] = distmat[q_num, indices[q_num, :]][mask]
                # gloabl id's of matched identities from gallery
                query_dict["matched_global_ids"] = np.array(
                    self.gallery_global_ids_list
                )[indices[q_num, :][mask]]
                # embeddinng of query
                query_dict["embedding"] = embeddings[q_num, :].cpu().numpy()
                # global_id of query
                query_dict["query_global_id"] = self.query_global_ids_list[q_num]
                # pack into an obejct
                out[query_path] = query_dict

                # if we dont match two global ids to one another (ie ocsort got it wrong and we have two global ids for one unique object)
                if (
                    np.array(self.gallery_global_ids_list)[indices[q_num, :][mask]][0]
                    != self.new_query_global_ids_list[q_num]
                ):
                    self.unique_query_global_ids.append(
                        self.new_query_global_ids_list[q_num]
                    )
                    self.unique_gallery_global_ids.append(
                        np.array(self.gallery_global_ids_list)[indices[q_num, :][mask]][
                            0
                        ]
                    )

                # this list contains all the new ids associated with the original query ids
                self.new_query_global_ids_list[q_num] = np.array(
                    self.gallery_global_ids_list
                )[indices[q_num, :][mask]][0]

        self.matched = out

        # print("\n\nOUT RESULT")
        # print(out)
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

        # two things need to happen 1) The index needs to be added (or resivour sampled) for an addition to the gallery data. Also, we then need to append the detection to the gallery_embeddings, the gallery_global_ids and the gallery image paths.

        # Also, we need to go find all the query images that did NOT have a match meeting the threshold and then subsequently add them to the gallery_embeddings, the gallery_global_ids and the gallery_image paths with their original global_ids which were either supplied by the tracker (ocsort) or by the nearest neighbour in terms of location."""
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

        # for all the query images that had no match whatsoever (meaning they are theoretically new detections), we need to add them to the embeddings, the image_paths, the global_id and finally the connecting df
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
        print("\n\nCONNECTING DICT")

        print(self.connecting_dict)

        return 0

    def reid(self, arg_queue, que, save_image_interval, find_similar_interval):
        """
        Applies re-identification (ReID) to track people across frames. At the moment, this is called out of the of the main branch to run in parallel.

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

        # Due to the function running in parallel, we want it to always keep running.
        while True:
            # if there is some info passed from the main script
            if not arg_queue.empty():

                time1 = time.time()
                (
                    frame_id,
                    frames,
                    boxes_with_probabilities,
                ) = arg_queue.get()
                print("\n\nthis is the time to get a heavy thing", time.time() - time1)

                if frame_id == 0:
                    # initialise
                    self.initialise(boxes_with_probabilities, frames)
                    print("done with 1")

                    que.put(1)

                if (
                    frame_id % save_image_interval == 0
                    and frame_id > 0
                    and frame_id % (find_similar_interval - save_image_interval) != 0
                    # and frame_id % (find_similar_interval - save_image_interval) != 0
                ):
                    print("\nstarting with saving images")

                    self.save_images(boxes_with_probabilities, frames)

                    print("\ndone with saving images")

                    que.put(1)

                if (frame_id % find_similar_interval == 0) and (frame_id > 0):

                    print("Started the finder")

                    # run the similarity function
                    self.embed_from_directory(gallery=False)

                    original_id_list, new_query_ids = self.find_similar_people()

                    # clear out the query info, as this is no longer needed
                    self.query_image_paths_list = []
                    self.query_global_ids_list = []

                    if original_id_list == new_query_ids:
                        print("\nidentical")

                    else:
                        print("\n replaced!!")

                    # this is the information that needs to be returned to the main branch of the programme
                    que.put(original_id_list)
                    que.put(new_query_ids)
                    # que.put(
                    #     (self.unique_gallery_global_ids, self.unique_query_global_ids)
                    # )

                else:
                    que.put(1)

    def reid_wrapper(
        self,
        all_boxes_with_probs,
        context_df,
        frame_id,
        arg_que,
        frames,
        multithread_que,
        REID_SAVE_IMAGE_INTERVAL,
        REID_FIND_SIMILAR_INTERVAL,
        view,
        # trajectories,
        done_flag,
    ):
        """contains the logic of multiprocessing"""
        # join the boxes with global ids and box
        for box in range(len(all_boxes_with_probs)):
            all_boxes_with_probs[box] = import_global_ids(
                context_df, all_boxes_with_probs[box]
            )

        if frame_id == 0:
            arg_que.put(
                (
                    frame_id,
                    frames,
                    all_boxes_with_probs,
                )
            )

            reid_process = mp.Process(
                target=self.reid,
                args=(
                    arg_que,
                    multithread_que,
                    REID_SAVE_IMAGE_INTERVAL,
                    REID_FIND_SIMILAR_INTERVAL,
                ),
                name="REID process",
            )
            reid_process.start()
            done_flag = False

        # print("\nbefore context:", context_df[context_df["cam_id"] == 2])

        # there is simething the reid model wants to give us (thus it has finished running):
        if not multithread_que.empty():

            # get whatever is in the que
            original = multithread_que.get()
            # 1 is the flag that is returned that shows that it is done with non-returning feedback
            if original != 1:

                new = multithread_que.get()
                # unique_global = multithread_que.get()

                # replace the global id in the context dataframe
                context_df["global_id"] = context_df["global_id"].replace(original, new)

                # replace the global id mapping method in the local to global table
                view.region.local_to_global["global_id"] = view.region.local_to_global[
                    "global_id"
                ].replace(original, new)

                # change the saved history by altering the trajectories.

                # for unique_replacement in range(len(unique_global[0])):

                #     trajectories[unique_global[0][unique_replacement]][0].extend(
                #         trajectories[unique_global[1][unique_replacement]][0]
                #     )
                #     trajectories[unique_global[0][unique_replacement]][1].extend(
                #         trajectories[unique_global[1][unique_replacement]][1]
                #     )

            # set the done flag, independependent of what the reid function did
            done_flag = True

        # check if the done flag is true
        if done_flag == True:
            # only pass the info to the reid model under certain conditions
            if frame_id % 5 == 0 and frame_id > 0:
                done_flag = False
                arg_que.put(
                    (
                        frame_id,
                        frames,
                        all_boxes_with_probs,
                    )
                )

        return (
            arg_que,
            context_df["global_id"],
            view.region.local_to_global["global_id"],
            # trajectories,
            done_flag,
        )
