import os
import subprocess

left_dir = '/home/umap3/calib/left'
right_dir = '/home/umap3/calib/right'

os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

left_count = len(os.listdir(left_dir))
right_count = len(os.listdir(right_dir))

print("Instructions:")
print("Press ENTER to capture LEFT camera image.")
print("Then press ENTER again to capture RIGHT camera image.")
print("Type 'q' and ENTER at any prompt to quit.")

while True:
    cmd = input("Capture LEFT camera? (ENTER to capture, 'q' to quit): ")
    if cmd.lower() == 'q':
        break
    left_filename = os.path.join(left_dir, f'left_{left_count:03d}.jpg')
    subprocess.run(['libcamera-still', '-o', left_filename, '--camera', '0', '-t', '1000'])
    print(f"Saved LEFT image: {left_filename}")
    left_count += 1

    cmd = input("Capture RIGHT camera? (ENTER to capture, 'q' to quit): ")
    if cmd.lower() == 'q':
        break
    right_filename = os.path.join(right_dir, f'right_{right_count:03d}.jpg')
    subprocess.run(['libcamera-still', '-o', right_filename, '--camera', '1', '-t', '1000'])
    print(f"Saved RIGHT image: {right_filename}")
    right_count += 1

print("Finished capturing images.")
