import os
import os.path as osp
import re

import numpy as np
from numpy import array, int32
from scipy.io import loadmat

from .base import BaseDataset


class MVN(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "MVN"
        self.img_prefix = osp.join(root, "frames")
        self.anno_preloaded = 'data/anno_loaded/{}.txt'.format(split)
        super(MVN, self).__init__(root, transforms, split)

    # def _get_cam_id(self, img_name):
    #     match = re.search(r"c\d", img_name).group().replace("c", "")
    #     return int(match)

    def _load_queries(self):
        if os.path.exists(self.anno_preloaded):
            print('Loading preloaded one...')
            with open(self.anno_preloaded, 'r') as fin:
                queries = eval(fin.read())
            pids = []
            for i in queries:
                pids.append(i['pids'][0])
            np.savetxt('data/anno_loaded/pids_{}.txt'.format(self.split), pids)
            return queries

        query_info = osp.join(self.root, "query_info.txt")
        with open(query_info, "rb") as f:
            raw = f.readlines()

        queries = []
        for line in raw:
            linelist = str(line, "utf-8").split(" ")
            pid = int(linelist[0])
            x, y, w, h = (
                float(linelist[1]),
                float(linelist[2]),
                float(linelist[3]),
                float(linelist[4]),
            )
            roi = np.array([x, y, x + w, y + h]).astype(np.int32)
            roi = np.clip(roi, 0, None)  # several coordinates are negative
            img_name = linelist[5].strip() + ".jpg"
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix.replace('frames', 'query_portraits'), img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([pid]),
                    # "cam_id": self._get_cam_id(img_name),
                }
            )
        print('-----------\n-----------\n{} pid:'.format(self.split))
        with open('data/anno_loaded/{}.txt'.format(self.split), 'w') as fout:
            fout.write(str(queries))
        pids = []
        for i in queries:
            pids.append(i['pids'][0])
        np.savetxt('data/anno_loaded/pids_{}.txt'.format(self.split), pids)
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "train":
            imgs = loadmat(osp.join(self.root, "frame_train.mat"))["img_index_train"]
        else:
            imgs = loadmat(osp.join(self.root, "frame_test.mat"))["img_index_test"]
        return [img[0][0] + ".jpg" for img in imgs]

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        if os.path.exists(self.anno_preloaded):
            print('Loading preloaded one...')
            with open(self.anno_preloaded, 'r') as fin:
                annotations = eval(fin.read())
            pids = []
            for i in annotations:
                pids.append(i['pids'][0])
            np.savetxt('data/anno_loaded/pids_{}.txt'.format(self.split), pids)
            return annotations

        annotations = []
        imgs = self._load_split_img_names()
        print('Loading annotations >>>')
        for idx_img, img_name in enumerate(imgs):
            if idx_img % 1000 == 0:
                print('{}/{},\t'.format(idx_img, len(imgs)), end='')
            anno_path = osp.join(self.root, "annotations", img_name.replace('.jpg', ''))
            anno = loadmat(anno_path)
            box_key = "box_new"
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            rois = np.round(anno[box_key][:, 1:] / 3)
            ids = anno[box_key][:, 0]
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)

            rois[:, 2:] += rois[:, :2]
            ids[ids == -2] = 5555  # assign pid = 5555 for unlabeled people
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": rois.astype(np.int32),
                    # FIXME: (training pids) 1, 2,..., 478, 480, 481, 482, 483, 932, 5555
                    "pids": ids.astype(np.int32),
                    # "cam_id": self._get_cam_id(img_name),
                }
            )
        with open('data/anno_loaded/{}.txt'.format(self.split), 'w') as fout:
            fout.write(str(annotations))
        print('\n<<< End')
        print('-----------\n-----------\n{} pid:'.format(self.split))
        pids = []
        for i in annotations:
            pids.append(i['pids'][0])
        np.savetxt('data/anno_loaded/pids_{}.txt'.format(self.split), pids)
        return annotations
