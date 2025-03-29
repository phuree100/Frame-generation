# Python script to generate frames.txt with frame names like frame_00010.png, frame_00015.png, etc.

def generate_frames(filename='frames.txt', start=10, step=5, end=3600):
    with open(filename, 'w') as file:
        for i in range(start, end + 1, step):
            # Format the frame number with leading zeros to match the format
            frame_name = f"file './frame_{i:05}.png'"
            file.write(frame_name + '\n')

# Call the function to generate frames.txt
generate_frames()