import os
from pathlib import Path
import xml.etree.ElementTree as ET
from loguru import logger

import cv2
import numpy as np
import random


from .datasets_wrapper import Dataset
from .voc import AnnotationTransform
from yolox.evaluators.voc_eval import voc_eval, det_image_visualization


def write_filesnames(filenames, path):
    num_files = 0
    with open(path, "w") as f:
        for filename in filenames:
            f.write(filename)
            f.write("\n")  # file is created with a final blank line
            num_files += 1
        f.close()
    logger.info(f"ImageSet file created in {path}: {num_files} lines written.")


def retrieve_annotation(data_dir, filename):
    # bad case: cannot find annotation file.
    annopath = os.path.join(data_dir, "Annotations", "{}.xml")
    return ET.parse(annopath.format(filename)).getroot()


def build_imageset(data_dir, set_type="all"):
    images_dir = os.path.join(data_dir, "Images")
    imageset_dir = os.path.join(data_dir, "ImageSets")
    Path(imageset_dir).mkdir(exist_ok=True)

    image_filenames = set()
    for image_file in Path(images_dir).iterdir():
        filename = image_file.with_suffix("").parts[-1]
        if set_type == "all":
            # just add all files; don't retrieve annotations
            image_filenames.add(filename)
            continue
        # retrieve respective annotation file to filter based on criteria
        root = retrieve_annotation(data_dir, filename)
        if set_type == "pure":  # skip all pseudo and bad
            if check_element(root, "pseudo", 1) or check_element(root, "bad", 1):
                continue
        else:
            if check_element(
                root, set_type, 0, default=True
            ):  # if unable to find, true by default
                continue
        image_filenames.add(filename)

    write_filesnames(image_filenames, os.path.join(imageset_dir, f"{set_type}.txt"))

    return image_filenames


def check_element(root, element, value, default=False):
    elem = root.find(element)
    if elem is not None:
        try:
            return int(elem) == value
        except:
            return default
    else:
        return default


EXPECTED_FILE_STRUCTURE = (
    "\ndata_dir\n"
    + "  - Annotations\n"
    + "    - \{img_id\}.xml files\n"
    + "  - Images\n"
    + "    - \{img_id\}.jpg files\n"
)


class CustomVOC(Dataset):
    """ YOLOX dataset class for custom build voc.
    Unlike official VOC, the expected file structure is as such:
    data_dir
        - Annotations
            - (img_id).xml files
        - Images
            - (img_id).jpg files
    """

    def __init__(
        self,
        data_dir,
        class_names,
        img_size=(512, 512),
        preproc=None,
        target_transform=None,
        dataset_name="CustomVOC",
        cache=False,
        remove_fraction=0.0,
        keep_difficult=True,
        keep_fake=True,
        keep_truncated=True,
        keep_occluded=True,
        set_type="all",
    ):
        super().__init__(
            img_size
        )  # Cannot inherit from VOCDet due to file struct ref in init.

        self.root = data_dir
        if target_transform is None:
            self.target_transform = AnnotationTransform(
                class_to_ind=dict(
                    [(label, index) for index, label in enumerate(class_names)]
                ),
                keep_difficult=keep_difficult,
                keep_fake=keep_fake,
                keep_occluded=keep_occluded,
                keep_truncated=keep_truncated,
            )
        else:
            self.target_transform = target_transform
        self.img_size = img_size
        self.preproc = preproc

        # dataset file details:
        self.class_names = class_names

        self._annopath = os.path.join(data_dir, "Annotations", "{}.xml")
        self._imgpath = os.path.join(data_dir, "Images", "{}.jpg")
        self._setpath = os.path.join(data_dir, "ImageSets", "all.txt")
        # TODO: add imagesets for annotations of different properties

        # check files:
        self._assert_folder_exists("Annotations")
        self._assert_folder_exists("Images")
        if not os.path.exists(self._setpath):  # Create if not exist.
            build_imageset(data_dir)  # FYI: used for validation.

        # Different compared to vars in VOCDet:
        self.name = dataset_name
        self.imgs = None
        if cache:
            self._cache_images()

        # populate the image ids for the dataset loader:
        self.ids = list()
        # TODO: use respective imageset file for image ids
        for filename in build_imageset(data_dir, set_type=set_type):
            if random.random() < remove_fraction:
                continue
            self.ids.append(filename)
        self.annotations = self._load_coco_annotations()

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x), range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath.format(img_id)).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath.format(img_id), cv2.IMREAD_COLOR)
        assert img is not None

        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        return img, target, img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        # need to overwrite super class method for this
        # the dataset does not support the requirements.

        # write detections to a file so they can be retrieved:
        self._write_voc_results_file(all_boxes)

        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        for iou in IouTh:
            mAP = self.calculate_mAP(iou_threshold=iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.class_names):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    # index = index[1] -> ids = img_id
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _get_voc_results_file_template(self):
        # similar to super but reference to year is removed.
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results")  # , "VOC", "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def visualize_detections(self):
        class_visualizations = {}

        for label in self.class_names:
            if label == "__background__":
                continue
            detpath = self._get_voc_results_file_template().format(label)

            class_visualizations[label] = det_image_visualization(
                detpath, self._imgpath, label
            )
        return class_visualizations

    def calculate_mAP(self, iou_threshold=0.5):
        imagesetfile = self._setpath
        cachedir = os.path.join(self.root, "annotations_cache")  # , "VOC", "train")
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)

        aps = []

        for label in self.class_names:
            if label == "__background__":
                continue

            # cool that this can be done (formating outside func call):
            filename = self._get_voc_results_file_template().format(label)
            rec, prec, ap = voc_eval(
                filename,
                self._annopath,
                imagesetfile,
                label,
                cachedir,
                ovthresh=iou_threshold,
            )

            aps += [ap]  # extend list

            if iou_threshold == 0.5:
                print("AP for {} = {:.4f}".format(label, ap))
        if iou_threshold == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))

        return np.mean(aps)

    def _assert_folder_exists(self, folder):
        assert os.path.exists(os.path.join(self.root, folder)), (
            f"Could not find {folder}. Check that the dataset file structure is correct:"
            + EXPECTED_FILE_STRUCTURE
        )

