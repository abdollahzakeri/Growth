import argparse
import os
import cv2
import numpy as np


def undistort_and_crop(
    input_folder: str,
    output_folder: str,
    k1: float = -0.060,
    k2: float = 0.0,
    fov_deg: float = 110.0,
    crop_x: int = 170,
    crop_y: int = 165,
    crop_w: int = 1580,
    crop_h: int = 697,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    D = np.array([k1, k2, 0, 0], dtype=np.float32)
    fov_rad = np.deg2rad(fov_deg)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG")
    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(exts):
            continue
        in_path = os.path.join(input_folder, fname)
        img = cv2.imread(in_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        fx = w / (2 * np.tan(fov_rad / 2))
        fy = fx
        cx = w / 2
        cy = h / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, K, D, None, new_K)

        x2 = min(crop_x + crop_w, undistorted.shape[1])
        y2 = min(crop_y + crop_h, undistorted.shape[0])
        cropped = undistorted[crop_y:y2, crop_x:x2]

        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, cropped)


def main():
    parser = argparse.ArgumentParser(description="Undistort and crop images for time-lapse dataset")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder for undistorted & cropped images")
    parser.add_argument("--k1", type=float, default=-0.060)
    parser.add_argument("--k2", type=float, default=0.0)
    parser.add_argument("--fov", type=float, default=110.0, help="Field of view in degrees")
    parser.add_argument("--crop_x", type=int, default=170)
    parser.add_argument("--crop_y", type=int, default=165)
    parser.add_argument("--crop_w", type=int, default=1580)
    parser.add_argument("--crop_h", type=int, default=697)
    args = parser.parse_args()

    undistort_and_crop(
        input_folder=args.input,
        output_folder=args.output,
        k1=args.k1,
        k2=args.k2,
        fov_deg=args.fov,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
    )


if __name__ == "__main__":
    main()


