# LiveBicepCounter

LiveBicepCounter is a Python-based application designed to count biceps curls in real-time using a webcam.
By leveraging computer vision technology with OpenCV and MediaPipe for pose estimation,
users can receive instant feedback on their workout progress with an on-screen display showing the rep count and a visual progress bar.

## Key Features:
- Real-time detection and counting of bicep curls
- Visual effects displaying rep count and progress
- Built with OpenCV for image processing and MediaPipe for accurate pose estimation
- User-friendly interface with keypress controls for interaction

## Getting Started

### Prerequisites:
Ensure you have Python installed on your computer along with the following packages:
- OpenCV (For image processing)
- NumPy (For numerical computations)
- MediaPipe (For pose estimation)

### Installation:
Install the necessary packages using pip by running the following command:
```bash
pip install opencv-python numpy mediapipe
```


Position yourself in front of the webcam and start performing bicep curls. The application will display the current rep count and a visual progress bar reflecting your form.

## Controls
'r' - Reset the rep count.
'q' - Quit the application.
## Contributions
Community contributions are welcome! For significant changes or enhancements, please open an issue first to discuss your ideas with the maintainers.


### TUTORIALS.md
```markdown
# Tutorials

## Getting Started

To use the LiveBicepCounter, position your webcam to capture your upper body, especially focusing on the arm you'll be using to perform bicep curls.

Ensure you perform the curls within the camera's view and maintain a consistent pace for the program to accurately count your reps.

The script will display a progress bar that fills up as you complete a bicep curl, along with rep count and percentage labels on the frame itself.

## Customization

If you would like to customize the detection or the visual feedback, you can modify the `poseestimationmodule.py` and `bicepApp.py` files to suit your preferences.
 You can adjust the following elements:
- Pose landmarks to suit different exercises or additional functionality
- Thresholds for rep counting logic
- Colors and styling of visual feedback elements like progress bars and labels

Make sure to understand the pose landmark IDs and relationships as defined by MediaPipe's Pose model when making such adjustments.
```



## License
LiveBicepCounter is available under the MIT License. For more information, consult the included LICENSE file in the repository.
