
import asyncio
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from docrag.core import llm_processing
from docrag.core.document_image_store import PageElement
from docrag.core.utils import pil_image_to_bytes
from docrag.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)



def get_doclayout_model(
    model_weights: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
):
    model_weights = Path(model_weights)
    if model_weights.exists():
        return model_weights

    model_weights = MODELS_DIR / "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_name = model_weights.name
    model_repo = "juliozhao/DocLayout-YOLO-DocStructBench"
    
    logger.info(f"Model repo: {model_repo}")
    logger.info(f"Model name: {model_name}")
    if (
        not model_weights.exists() or not model_weights.is_file()
    ):  # Check if dir exists and is not empty
        logger.info(
            f"Model not found locally. Downloading from Hugging Face Hub: {model_repo}"
        )
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=model_name,
                local_dir=MODELS_DIR,
            )
            logger.info("Model download complete.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return model_weights

    return model_weights



class PageElementType(Enum):
    """Type of element."""

    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    TEXT = "text"
    TITLE = "title"
    TABLE_CAPTION = "table_caption"
    FORMULA_CAPTION = "formula_caption"
    TABLE_FOOTNOTE = "table_footnote"
    FIGURE_CAPTION = "figure_caption"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_doclayout_yolo(cls, element_type: str):
        if element_type == "figure":
            return cls.FIGURE
        elif element_type == "table":
            return cls.TABLE
        elif element_type == "isolate_formula":
            return cls.FORMULA
        elif element_type == "plain text":
            return cls.TEXT
        elif element_type == "title":
            return cls.TITLE
        elif element_type == "table_caption":
            return cls.TABLE_CAPTION
        elif element_type == "formula_caption":
            return cls.FORMULA_CAPTION
        elif element_type == "figure_caption":
            return cls.FIGURE_CAPTION
        elif element_type == "table_footnote":
            return cls.TABLE_FOOTNOTE
        elif element_type == "abandon":
            return cls.UNKNOWN
        else:
            return cls.UNKNOWN

# @dataclass
# class DocLayoutExtractionResults:
#     element_type: PageElementType
#     image: Image.Image
#     confidence: float
#     bbox: tuple[int, int, int, int]
#     sort_order: int

def extract_page_elements_with_yolo(
        images: Union[str, Path, Image.Image] | List[Image.Image],
        model_weights="doclayout_yolo_docstructbench_imgsz1024.pt",
        image_size=1024,
        confidence_threshold=0.2,
        device="cpu",
    ) -> List[PageElement]:
        """Process each detection and create element info."""
        if isinstance(images, str) or isinstance(images, Path):
            images = [Image.open(Path(images))]

        model_weights = get_doclayout_model(model_weights)

        model = YOLOv10(model_weights)
        # Run YOLO prediction
        doclayout_results = model.predict(
            images,
            imgsz=image_size,
            conf=confidence_threshold,
            device=device,
        )
        
        processed_results = []
        for doclayout_result in doclayout_results:
            results = doclayout_result[0]
            image = Image.fromarray(results.orig_img)
            boxes = results.boxes
            class_names = results.names
            class_name_map = {i: class_name for i, class_name in class_names.items()}

            if boxes is None or len(boxes) == 0:
                logger.info("No detections found")
                elements = []
                return elements

            original_width, original_height = image.size

            elements = []
            for i, box in enumerate(boxes.xyxy):
                
                
                # Get class information
                cls_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                element_type = PageElementType.from_doclayout_yolo(class_name_map[cls_id])

                # Extract and validate bounding box coordinates
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(original_width, int(x2))
                y2 = min(original_height, int(y2))

                # Crop the element
                cropped_element = image.crop((x1, y1, x2, y2))

                # Create element info
                element = PageElement(
                    element_type=element_type.value,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    image=pil_image_to_bytes(cropped_element),
                )

                # Store the element
                elements.append(element)
                
            # elements = match_captions_and_footnotes(elements)
            elements = sort_image_elements(elements)
            
            processed_results.append(elements)

        return processed_results


def find_nearest_neighbor_element(
    element: PageElement,
    neighbor_elements: List[PageElement],
    max_distance: float = 100,
) -> Optional[PageElement]:
    """Find the nearest caption to a given element based on proximity and vertical alignment."""

    if len(neighbor_elements) == 0:
        return None, neighbor_elements

    best_neighbor_element = None
    best_score = float("inf")

    element_box = element.bbox

    element_center_x = (element_box[0] + element_box[2]) / 2
    element_left = element_box[0]
    element_right = element_box[2]
    element_top = element_box[1]
    element_bottom = element_box[3]

    for i, neighbor_element in enumerate(neighbor_elements):
        neighbor_element_box = neighbor_element.bbox
        neighbor_element_left = neighbor_element_box[0]
        neighbor_element_right = neighbor_element_box[2]
        neighbor_element_top = neighbor_element_box[1]
        neighbor_element_bottom = neighbor_element_box[3]

        # Calculate vertical distance (prefer captions below the element)
        element_top_neighbor_element_bottom = abs(neighbor_element_bottom - element_top)

        # Calculate vertical distance (prefer captions above the element)
        element_bottom_neighbor_element_top = abs(neighbor_element_top - element_bottom)

        smallest_vertical_distance = min(
            element_top_neighbor_element_bottom, element_bottom_neighbor_element_top
        )
        if smallest_vertical_distance < best_score:
            best_score = smallest_vertical_distance
            best_neighbor_element = neighbor_element
            best_neighbor_element_index = i

    neighbor_elements.pop(best_neighbor_element_index)
    return best_neighbor_element, neighbor_elements

def match_captions_and_footnotes(elements: Dict[PageElementType, List[PageElement]]):
    """Find and store caption and footnote boxes."""

    table_captions = []
    table_footnotes = []
    figure_captions = []
    formula_captions = []

    new_elements = []
    # Separate caption and footnote elements
    for element in elements:
        if element.element_type == PageElementType.TABLE_CAPTION.value:
            table_captions.append(element)
        elif element.element_type == PageElementType.TABLE_FOOTNOTE.value:
            table_footnotes.append(element)
        elif element.element_type == PageElementType.FIGURE_CAPTION.value:
            figure_captions.append(element)
        elif element.element_type == PageElementType.FORMULA_CAPTION.value:
            formula_captions.append(element)
        else:
            new_elements.append(element)
    # Match captions and footnotes to elements
    for element in new_elements:
        if element.element_type == PageElementType.TABLE.value:
            best_caption_neighbor_element, table_captions = (
                find_nearest_neighbor_element(element, table_captions)
            )
            best_footnote_neighbor_element, table_footnotes = (
                find_nearest_neighbor_element(element, table_footnotes)
            )
            element.caption = best_caption_neighbor_element
            element.footnote = best_footnote_neighbor_element
        elif element.element_type == PageElementType.FORMULA.value:
            best_caption_neighbor_element, formula_captions = (
                find_nearest_neighbor_element(element, formula_captions)
            )
        elif element.element_type == PageElementType.FIGURE.value:
            best_caption_neighbor_element, figure_captions = (
                find_nearest_neighbor_element(element, figure_captions)
            )
            element.caption = best_caption_neighbor_element

    return new_elements

def sort_image_elements(elements):
    """Sort elements by logical reading order."""
    sort_indices = find_element_logical_reading_order(elements)
    elements = [elements[i] for i in sort_indices]
    return elements

def projection_by_bboxes(boxes: np.array, axis: int) -> np.ndarray:
    """
    Args:
        boxes: [N, 4]
        axis: 

    Returns:
        1D 

    """
    assert axis in [0, 1]
    length = np.max(boxes[:, axis::2])
    res = np.zeros(length, dtype=int)
    # TODO: how to remove for loop?
    for start, end in boxes[:, axis::2]:
        res[start:end] += 1
    return res




def split_projection_profile(arr_values: np.array, min_value: float, min_gap: float):
    """Split projection profile:

    ```
                              ┌──┐
         arr_values           │  │       ┌─┐───
             ┌──┐             │  │       │ │ |
             │  │             │  │ ┌───┐ │ │min_value
             │  │<- min_gap ->│  │ │   │ │ │ |
         ────┴──┴─────────────┴──┴─┴───┴─┴─┴─┴───
         0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    ```

    Args:
        arr_values (np.array): 1-d array representing the projection profile.
        min_value (float): Ignore the profile if `arr_value` is less than `min_value`.
        min_gap (float): Ignore the gap if less than this value.

    Returns:
        tuple: Start indexes and end indexes of split groups.
    """
    # all indexes with projection height exceeding the threshold
    arr_index = np.where(arr_values > min_value)[0]
    if not len(arr_index):
        return

    # find zero intervals between adjacent projections
    # |  |                    ||
    # ||||<- zero-interval -> |||||
    arr_diff = arr_index[1:] - arr_index[0:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]
    arr_zero_intvl_start = arr_index[arr_diff_index]
    arr_zero_intvl_end = arr_index[arr_diff_index + 1]

    # convert to index of projection range:
    # the start index of zero interval is the end index of projection
    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
    arr_end += 1  # end index will be excluded as index slice

    return arr_start, arr_end

def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]):
    """

    Args:
        boxes: (N, 4)
        indices: 
        res: 

    """

    assert len(boxes) == len(indices)

    _indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[_indices]
    y_sorted_indices = indices[_indices]



    y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    pos_y = split_projection_profile(y_projection, 0, 1)
    if not pos_y:
        return

    arr_y0, arr_y1 = pos_y
    for r0, r1 in zip(arr_y0, arr_y1):

        _indices = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)

        y_sorted_boxes_chunk = y_sorted_boxes[_indices]
        y_sorted_indices_chunk = y_sorted_indices[_indices]

        _indices = y_sorted_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_sorted_boxes_chunk[_indices]
        x_sorted_indices_chunk = y_sorted_indices_chunk[_indices]


        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        pos_x = split_projection_profile(x_projection, 0, 1)
        if not pos_x:
            continue

        arr_x0, arr_x1 = pos_x
        if len(arr_x0) == 1:
 
            res.extend(x_sorted_indices_chunk)
            continue

        for c0, c1 in zip(arr_x0, arr_x1):
            _indices = (c0 <= x_sorted_boxes_chunk[:, 0]) & (
                x_sorted_boxes_chunk[:, 0] < c1
            )
            recursive_xy_cut(
                x_sorted_boxes_chunk[_indices], x_sorted_indices_chunk[_indices], res
            )

def find_element_logical_reading_order(elements: List[PageElement]) -> List[int]:
    """Sort elements by logical reading order: multi-column elements by x then y,
    with full-width elements inserted based on y-coordinate."""
    from scipy.cluster.vq import kmeans, vq
    if not elements:
        return []

    # Calculate center coordinates and width for each element
    element_bboxes=[]
    for i, element in enumerate(elements):
        element_bboxes.append(element.bbox)

    res=[]
    recursive_xy_cut(np.array(element_bboxes), np.arange(len(element_bboxes)), res)
    return res
