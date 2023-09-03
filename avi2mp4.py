import cv2

def convert_avi_to_mp4(input_path, output_path, fps=10.0):
    try:
        # Read the input AVI video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Convert each frame from AVI to MP4
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Release the resources
        cap.release()
        out.release()
        
        print(f"Conversion successful! MP4 video saved at {output_path}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        
        
def main():
    convert_avi_to_mp4(r'C:\Users\vlado\OneDrive\Desktop\my_video_12.avi', r'C:\Users\vlado\OneDrive\Desktop\my_video_12.mp4')
    
    
if __name__ == "__main__":
    main()