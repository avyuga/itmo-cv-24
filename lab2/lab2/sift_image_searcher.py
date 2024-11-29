from image_searcher import ImageSearcher
import numpy as np
import cv2


class SiftImageSearcher(ImageSearcher):
    def find_image(
        self,
        haystack: np.ndarray,
        needle: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """
        Find the needle image within the haystack image using SIFT features.
        Args:
            haystack: The larger image to search in (numpy array)
            needle: The smaller image to find (numpy array)

        Returns:
            Tuple of (x, y, width, height) representing the matching rectangle,
            or None if no match is found
        """
        # Initialize SIFT detector
        sift = cv2.SIFT.create()

        # Find keypoints and descriptors for both images
        kp1, des1 = sift.detectAndCompute(needle, None)
        kp2, des2 = sift.detectAndCompute(haystack, None)

        # If no features found, return None
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None

        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Find matches
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test to find good matches
        filtered_matches = []
        for nearest_match, second_nearest_match in matches:
            if nearest_match.distance < 0.7 * second_nearest_match.distance:
                filtered_matches.append(nearest_match)

        # Need at least 4 good matches to find homography
        if len(filtered_matches) < 4:
            return None

        # Get coordinates of matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

        # Find homography matrix
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        # Get corners of needle image
        h, w = needle.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # Transform corners to find location in haystack
        transformed = cv2.perspectiveTransform(corners, H)

        # Get bounding rectangle
        x = int(min(transformed[:, 0, 0]))
        y = int(min(transformed[:, 0, 1]))
        w = int(max(transformed[:, 0, 0]) - x)
        h = int(max(transformed[:, 0, 1]) - y)

        return (x, y, w, h)
