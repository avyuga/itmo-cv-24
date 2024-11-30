from image_searcher import ImageSearcher
import numpy as np


class TemplateImageSearcher(ImageSearcher):
    def __init__(self, step: int = 1, threshold: float = 0.01):
        """
        Initialize the TemplateImageSearcher.

        Args:
            step: Step size for sliding window iteration (default=1).
                 Higher values will search fewer positions but be faster.
        """
        self.step = max(1, step)
        self.threshold = threshold

    def find_image(
        self,
        haystack: np.ndarray,
        needle: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """
        Find the needle image within the haystack image.

        Args:
            haystack: The larger image to search in (numpy array)
            needle: The smaller image to find (numpy array)

        Returns:
            Tuple of (x, y, width, height) representing the matching rectangle,
            or None if no match is found
        """
        # Obtain the dimensions of the needle and haystack images
        needle_height, needle_width = needle.shape[:2]
        haystack_height, haystack_width = haystack.shape[:2]

        # Check if the needle is larger than the haystack
        if needle_height > haystack_height or needle_width > haystack_width:
            return None

        # Create a result matrix
        result = np.full((haystack_height - needle_height + 1,
                           haystack_width - needle_width + 1), np.inf)

        # Iterate over the haystack with a sliding window
        for y in range(0, result.shape[0], self.step):
            for x in range(0, result.shape[1], self.step):
                # Extract the current window
                window = haystack[y:y+needle_height, x:x+needle_width]
                # Calculate the sum of squared differences
                diff = np.sum((window - needle) ** 2) / (needle_height * needle_width)
                result[y, x] = diff

        if np.min(result) / 255.0 > self.threshold:
            return None

        # Get the position of the best match
        y, x = np.unravel_index(np.argmin(result), result.shape)

        return (x, y, needle_width, needle_height)
