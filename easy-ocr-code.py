import os
import glob
import easyocr
import cv2

# --- Configuration ---
IMAGE_DIR = "images"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "ocr_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load EasyOCR reader ---
reader = easyocr.Reader(['en'])

# --- Collect images ---
image_paths = (
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

if not image_paths:
    print(f"No images found in '{IMAGE_DIR}'.")
else:
    print(f"Found {len(image_paths)} image(s).\n")

# --- Process each image ---
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIP] Could not load: {image_path}")
        continue

    print(f"Processing: {image_path}")

    ocr_results = reader.readtext(image_path)

    if not ocr_results:
        print("  No text detected.\n")
        continue

    for (bbox, text, confidence) in ocr_results:
        print(f"  Text: {text!r:30s}  Confidence: {confidence:.2f}")

        # Draw bounding box
        pts = [(int(x), int(y)) for x, y in bbox]
        cv2.polylines(image, [__import__('numpy').array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw text label above the bounding box
        label = f"{text} ({confidence:.2f})"
        top_left = pts[0]
        cv2.putText(
            image, label,
            (top_left[0], top_left[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2,
        )

    # Save annotated image
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, image)
    print(f"  Saved annotated image: {out_path}\n")

print("Done.")
