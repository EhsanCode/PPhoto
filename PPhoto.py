import cv2
import argparse

DV_IMAGE_SIZE = 601
SIZE_RATIO = 1
HEAD_RATIO = 0.45
TOP_RATIO = 0.22
SIDES_RATIO = (SIZE_RATIO - HEAD_RATIO) / 2


def crop_image_around_face(img, face_loc):
    """
    Crop the image around a detected face.
    """
    head_size = face_loc[2]
    image_size = int(head_size / HEAD_RATIO)
    top_size = int(image_size * TOP_RATIO)
    sides_size = int(image_size * SIDES_RATIO)
    x = face_loc[0] - sides_size
    y = face_loc[1] - top_size
    w = image_size
    h = image_size
    dv_image = img[y:y + h, x:x + w, :]
    dv_image = cv2.resize(dv_image, (DV_IMAGE_SIZE, DV_IMAGE_SIZE))
    return dv_image


def detect_faces(img):
    """
    Detect faces in the given image.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces


def show_detected_faces(img, faces):
    """
    Display the image with rectangles around detected faces.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_images(img, faces, output_dir):
    """
    Save cropped face images to the output directory.
    """
    for index, face in enumerate(faces):
        dv_image = crop_image_around_face(img, face)
        output_path = f"{output_dir}/DV_{index}.jpg"
        print(f"Saved: {output_path}")
        cv2.imwrite(output_path, dv_image)


def arg_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Crop and save faces from an image")
    parser.add_argument("image_path", type=str, help="Path to the original image file")
    parser.add_argument("output_path", type=str, help="Path to save the generated images")
    return parser.parse_args()


def main():
    args = arg_parser()
    img = cv2.imread(args.image_path)
    faces = detect_faces(img)
    save_images(img, faces, args.output_path)
    # show_detected_faces(img, faces)


if __name__ == '__main__':
    main()
