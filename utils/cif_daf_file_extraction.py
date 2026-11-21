import os
from multiprocessing import Process
from typing import NoReturn

import bioformats
import bioformats.formatreader
import h5py
import javabridge
import numpy as np
from tqdm import tqdm


def save_image_as_h5(reader: bioformats.ImageReader, id: int, file_path: str) -> int:
    """
    Save multichannel image and its corresponding mask from a CIF file to HDF5 format.

    This function reads both the image and its mask and saves them in a single HDF5 file.
    The data is stored in channels-first format (C, H, W) with GZIP compression.

    Args:
        reader (bioformats.ImageReader): Initialized ImageReader instance for accessing CIF data.
        id (int): Base series ID for the image in the CIF file. Mask is assumed to be at id + 1.
        file_path (str): Directory path where the HDF5 file will be saved.
    """
    output_file = os.path.join(file_path, f"image_{id // 2}.h5")

    try:
        image = np.transpose(reader.read(series=id, rescale=False), (2, 0, 1))
        mask = np.transpose(reader.read(series=id + 1, rescale=False), (2, 0, 1))

        if image is None or mask is None:
            raise ValueError("Failed to read image or mask data")

        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("image", data=image)
            hf.create_dataset("mask", data=mask)

        return 0

    except Exception as e:
        raise IOError(f"Failed to save H5 file: {str(e)}")


def extract_images(raw_path: str, img_folder_path: str) -> NoReturn:
    """
    Process a CIF file to extract images and masks, saving them as HDF5 files.

    This function initializes a Java virtual machine, reads the CIF file using bioformats,
    and extracts all image-mask pairs, saving them as separate HDF5 files.

    Args:
        raw_path (str): Full path to the input CIF file.
        img_folder_path (str): Base directory where extracted images will be saved.
    """
    print(f"Processing file: {raw_path}")
    results_path = os.path.join(
        img_folder_path, os.path.split(raw_path)[1].replace(" ", "_").split(".")[0]
    )
    if not os.path.isdir(results_path):
        print(f"Creating directory for extracted images at: {results_path}")
        os.makedirs(results_path)
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)

    try:

        reader = bioformats.ImageReader(path=raw_path)
        metadata = bioformats.get_omexml_metadata(raw_path)
        xml_reader = bioformats.OMEXML(metadata)
        image_count = xml_reader.get_image_count()
        print(f"Total image-mask pairs to process: {image_count // 2}")
        with tqdm(
            total=image_count // 2, desc="Processing images", unit="pair"
        ) as pbar:
            for i in range(0, image_count, 2):
                save_image_as_h5(reader, i, results_path)
                pbar.update(1)

        print(f"Successfully saved dataset as HDF5 files in: {results_path}")

    finally:
        javabridge.kill_vm()


def extract_from_CIF(cif_path: str, images_target_dir: str) -> NoReturn:
    """
    Extract images and masks from a CIF file in a separate process.

    This function creates a new process to handle the extraction, ensuring proper
    resource management and isolation.

    Args:
        cif_path (str): Full path to the input CIF file.
        images_target_dir (str): Base directory where extracted images will be saved.

    Note:
        - Runs extraction in a separate process to handle Java VM properly.
        - Waits for the process to complete before returning.
        - Any exceptions in the child process will be propagated to the parent.
    """
    p = Process(target=extract_images, args=(cif_path, images_target_dir))
    p.start()
    p.join()
