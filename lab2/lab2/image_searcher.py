from typing import Protocol
import numpy as np


class ImageSearcher(Protocol):
    """Protocol for searching one image within another."""

    def find_image(
        self,
        haystack: np.ndarray,
        needle: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """
        Find the needle image within the haystack image.

        Args:
            haystack: The larger image to search in (numpy array / cv2 image)
            needle: The smaller image to find (numpy array / cv2 image)

        Returns:
            Tuple of (x, y, width, height) representing the matching rectangle,
            or None if no match is found
        """
        ...
