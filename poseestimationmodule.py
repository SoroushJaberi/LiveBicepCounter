import cv2
import mediapipe as mp
import math


class poseDetector():
    """
    Class for detecting human poses in images or video frames using mediapipe.

    Attributes:
        mode (bool): If set to False, the solution treats the input images as a video stream.
        mcomplexity (int): Complexity of the pose landmark model: 0, 1 or 2.
        slandmarks (bool): If True, also extracts smoothed landmark information.
        esegmentation (bool): If True, enables selfie segmentation solution.
        ssegmentation (bool): If False, disables segmenting the pose.
        detconfidence (float): Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful.
        trackconfidence (float): Minimum confidence value ([0.0, 1.0]) for the tracking to be considered successful.
    """

    def __init__(self, mode=False, mcomplexity=1, slandmarks=True, esegmentation=False,
                 ssegmentation=True, detconfidence=0.5, trackconfidence=0.5):
        # Initialize the poseDetector with the given parameters
        self.mode = mode
        self.mcomplexity = mcomplexity
        self.slandmarks = slandmarks
        self.esegmentation = esegmentation
        self.ssegmentation = ssegmentation
        self.detconfidence = detconfidence
        self.trackconfidence = trackconfidence

        # Initialize mediapipe drawing utilities
        self.mpDraw = mp.solutions.drawing_utils
        # Initialize the mediapipe Pose solution
        self.mpPose = mp.solutions.pose
        # Set up the Pose model based on the provided parameters
        self.pose = self.mpPose.Pose(self.mode, self.mcomplexity, self.slandmarks,
                                     self.esegmentation, self.ssegmentation, self.detconfidence, self.trackconfidence)

    def findPose(self, img, draw=True):
        """
        Processes an image to detect the human pose and optionally draw the landmarks on the image.

        Parameters:
            img: The input image.
            draw (bool): If True, draw the landmarks on the image.

        Returns:
            An image with or without the drawn landmarks.
        """
        # Convert the image from BGR to RGB for processing
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image to find pose landmarks
        self.results = self.pose.process(imgRGB)

        # Check if any pose landmarks were found
        if self.results.pose_landmarks:
            # Draw pose landmarks on the image if specified
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        """
        Finds the position of each pose landmark on the image.

        Parameters:
            img: The input image.
            draw (bool): If True, draw the landmarks on the image.

        Returns:
            A list of lists containing the landmark id and its x, y coordinates.
        """
        # Initialize an empty list to store landmark information
        self.lmList = []
        # Check if any pose landmarks were found
        if self.results.pose_landmarks:
            # Iterate over all landmarks to get their positions
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # Get the shape of the input image
                # Calculate pixel coordinates of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Add the landmark id and its pixel coordinates to the list
                self.lmList.append([id, cx, cy])
                # Draw the landmarks on the image if specified
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three points (usually corresponding to joints).

        Parameters:
            img: The image where the points are to be drawn.
            p1, p2, p3: ids of the three points.
            draw (bool): If True, draw the points and the angle on the image.

        Returns:
            The calculated angle between the points.
        """
        # Extract the landmark coordinates from the list
        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]
        _, x3, y3 = self.lmList[p3]

        # Use arctan2 to calculate the angle between the points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # Adjust the angle if it is negative
        if angle < 0:
            angle += 360

        if draw:
            # Draw visual representation of the angle on the image
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)

            # Display the angle value near the second point
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 - 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


