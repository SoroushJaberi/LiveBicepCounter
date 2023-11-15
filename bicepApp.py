import cv2
import time
import numpy as np
import poseestimationmodule as pm

def draw_label(frame, text, pos, font_scale, thickness, color, bg_color, padding=10):
    """
    Draws a label with text, position, font scale, thickness, color, background color, and padding on the frame.

    Parameters:
        frame: The image onto which to draw the label.
        text: The string text for the label.
        pos: The position (x,y) to draw the label.
        font_scale: Font scale (size) for the text.
        thickness: Thickness of the text font.
        color: Color of the text.
        bg_color: Color of the label background.
        padding: Padding around the text inside the label.
    """
    # Calculate text size in pixels
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos  # Unpack the position coordinates
    # Draw background rectangle for label
    cv2.rectangle(frame, (x - padding, y + padding), (x + text_width + padding, y - text_height - baseline - padding),
                  bg_color, -1)
    # Draw border rectangle for label
    cv2.rectangle(frame, (x - padding, y + padding), (x + text_width + padding, y - text_height - baseline - padding),
                  color, 2)
    # Draw the text on the frame, which is now backed by the label
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_progress_bar(frame, value, max_value, pos, size, color, thickness):
    """
    Draws a horizontal progress bar on the frame.

    Parameters:
        frame: The image onto which to draw the progress bar.
        value: The current value representing the filled portion of the bar.
        max_value: The maximum value that the bar can represent.
        pos: The position (x,y) where the progress bar starts.
        size: The width in pixels of the progress bar.
        color: Color of the progress bar.
        thickness: Thickness (height) in pixels of the progress bar.
    """
    # Calculate the width of the bar that should be filled based on the value
    filled_part = int((value / max_value) * size)
    # Draw the outline of the progress bar
    cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + size, pos[1] + thickness), color, 2)
    # Fill the progress inside the bar based on the current value
    cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + filled_part, pos[1] + thickness), color, -1)

def resize_frame(frame, width=800, height=600):
    """
    Resizes the frame to the given width and height.

    Parameters:
        frame: The image to resize.
        width: Desired width of the frame.
        height: Desired height of the frame.

    Returns:
        The resized image.
    """
    # Resize the frame to the specified width and height using interpolation for good quality
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def main():
    """
    Main function to run pose estimation on a video feed and display information labels and a progress bar.
    """
    # Start video capture using the default camera
    cap = cv2.VideoCapture(0)
    # Set camera resolution to 800x600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    # Create a pose detector from the custom module
    detector = pm.poseDetector()
    dir = 0  # Direction flag for repetitions
    count = 0  # Counter for repetitions
    ptime = 0  # Previous time for FPS calculation

    # Define color scheme for the layout
    primary_color = (255, 100, 100)
    secondary_color = (100, 255, 100)
    background_color = (50, 50, 50)
    reset_button_color = (80, 130, 255)

    while True:
        ret, frame = cap.read()
        if ret:
            # Process the frame to detect pose and find landmarks
            frame = detector.findPose(frame, True)
            lmList = detector.findPosition(frame, False)
            # Resize the frame to the desired size
            frame_resized = resize_frame(frame)

            if len(lmList) != 0:
                # Detect angle, percentage and draw progress bar accordingly (Update indices as needed)
                angle = detector.findAngle(frame, 12, 14, 16, False)
                per = np.interp(angle, (210, 310), (0, 100))
                progress_bar_fill = np.interp(per, (0, 100), (0, 540))

                draw_progress_bar(frame_resized, progress_bar_fill, 540, (30, 560), 540, secondary_color, 20)

                # Change the color of the percentage label based on the value
                color = primary_color if per <= 50 else secondary_color
                # Update the repetition count based on the percentage completion
                if per == 100 and dir == 0:
                    count += 0.5
                    dir = 1
                if per == 0 and dir == 1:
                    count += 0.5
                    dir = 0

                # Draw the repetitions and percentage labels on the frame
                draw_label(frame_resized, f'{int(count)} Reps', pos=(580, 520), font_scale=2, thickness=2,
                           color=secondary_color, bg_color=primary_color)
                draw_label(frame_resized, f'{int(per)}%', pos=(10, 60), font_scale=2, thickness=2, color=color,
                           bg_color=background_color)

            # Calculate and display the FPS on the resized frame
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            draw_label(frame_resized, f'FPS: {int(fps)}', pos=(660, 40), font_scale=1, thickness=1,
                       color=secondary_color, bg_color=primary_color)

            # Display a reset button label on the frame
            draw_label(frame_resized, 'Reset Count (r)', pos=(30, 520), font_scale=1, thickness=1,
                       color=(255, 255, 255), bg_color=reset_button_color)

            # Show the final frame to the user
            cv2.imshow("Video", frame_resized)

            # Wait for either the 'r' key to reset the count or 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('r'):
                count = 0
                dir = 0  # Also reset direction when count is reset

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()