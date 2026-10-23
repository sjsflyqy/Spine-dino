import cv2
import numpy as np
from PIL import Image
from transformers import BitImageProcessor

class CustomBitImageProcessor(BitImageProcessor):
    def __init__(self, do_hist_equalization=False, **kwargs):
        super().__init__(**kwargs)
        self.do_hist_equalization = do_hist_equalization

    def apply_histogram_equalization(self, image):
        """
        Apply histogram equalization to an image.
        """
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # Grayscale
            equalized = cv2.equalizeHist(image_array)
        else:  # RGB
            channels = cv2.split(image_array)
            equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
            equalized = cv2.merge(equalized_channels)
        return Image.fromarray(equalized)

    def preprocess(self, images, **kwargs):
        """
        Apply custom preprocessing, including histogram equalization.
        """
        if self.do_hist_equalization:
            if isinstance(images, list):
                images = [self.apply_histogram_equalization(image) for image in images]
            else:
                images = self.apply_histogram_equalization(images)
        return super().preprocess(images, **kwargs)
