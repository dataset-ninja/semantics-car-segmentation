import supervisely as sly
import os
from collections import defaultdict
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
from supervisely.io.json import load_json_file
import shutil

from tqdm import tqdm


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(
                        team_id, teamfiles_path, local_path, progress_cb=pbar
                    )

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = os.path.join("car-segmentation", "images")
    anns_path = os.path.join("car-segmentation", "masks.json")
    batch_size = 10
    ds_name = "ds"

    def create_ann(image_path):
        labels = []

        image_name = get_file_name(image_path)
        img_height = name_to_shape[image_name]["height"]
        img_wight = name_to_shape[image_name]["width"]

        curr_image_data = name_to_data[image_name]
        for info in curr_image_data:
            obj_class = meta.get_obj_class(info[0])
            exterior = []
            for coords in info[2]:
                exterior.append([coords["y"], coords["x"]])
            poligon = sly.Polygon(exterior)
            label_poly = sly.Label(poligon, obj_class)
            labels.append(label_poly)

            left = info[1]["left"]
            right = left + info[1]["width"]
            top = info[1]["top"]
            bottom = top + info[1]["height"]
            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            label = sly.Label(rectangle, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_class_car = sly.ObjClass("car", sly.AnyGeometry, color=(0, 255, 255))
    obj_class_wheel = sly.ObjClass("wheel", sly.AnyGeometry, color=(128, 0, 0))
    obj_class_lights = sly.ObjClass("lights", sly.AnyGeometry, color=(255, 255, 0))
    obj_class_window = sly.ObjClass("window", sly.AnyGeometry, color=(128, 128, 0))

    project = api.project.create(
        workspace_id, project_name, change_name_if_conflict=True
    )
    meta = sly.ProjectMeta(
        obj_classes=[obj_class_car, obj_class_wheel, obj_class_lights, obj_class_window]
    )
    api.project.update_meta(project.id, meta.to_json())

    ann_data = load_json_file(anns_path)["assets"]

    name_to_data = defaultdict(list)
    name_to_shape = {}

    for _, image_data in ann_data.items():
        im_name = get_file_name(image_data["asset"]["name"])
        im_shape = image_data["asset"]["size"]
        name_to_shape[im_name] = im_shape
        regions = image_data["regions"]
        for region in regions:
            name_to_data[im_name].append(
                [region["tags"][0], region["boundingBox"], region["points"]]
            )

    images_names = os.listdir(images_path)

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        images_pathes_batch = [
            os.path.join(images_path, image_name) for image_name in img_names_batch
        ]

        img_infos = api.image.upload_paths(
            dataset.id, img_names_batch, images_pathes_batch
        )
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(img_names_batch))

    return project
