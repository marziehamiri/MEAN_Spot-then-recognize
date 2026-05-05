import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time


# ----------------------
# Convert polar → cartesian
# ----------------------
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# ----------------------
# Compute optical strain
# ----------------------
def computeStrain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis=1)
    v_y = v - pd.DataFrame(v).shift(-1, axis=0)
    u_y = u - pd.DataFrame(u).shift(-1, axis=0)
    v_x = v - pd.DataFrame(v).shift(-1, axis=1)

    o_s = np.array(np.sqrt(u_x**2 + v_y**2 + 0.5 * (u_y + v_x)**2).ffill(1).ffill(0))
    return o_s


# ----------------------
# MediaPipe eyebrow ROI
# ----------------------
def get_eyebrow_rect(img):

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, _ = img.shape

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    )
    res = face_mesh.process(img)

    if not res.multi_face_landmarks:
        return None

    face = res.multi_face_landmarks[0]

    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]

    pts = []

    for group in [LEFT_EYEBROW, RIGHT_EYEBROW]:
        for idx in group:
            pts.append((int(face.landmark[idx].x * w),
                        int(face.landmark[idx].y * h)))

    pts = np.array(pts)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    x_min = max(x_min - 2, 0)
    y_min = max(y_min - 2, 0)
    x_max = min(x_max + 2, w)
    y_max = min(y_max + 2, h)

    return x_min, y_min, x_max, y_max


# ----------------------
# Optical Flow + eyebrow ROI
# ----------------------
def preprocess_mediapipe(img1, img2, rect):

    x1, y1, x2, y2 = rect

    roi1 = img1[y1:y2, x1:x2]
    roi2 = img2[y1:y2, x1:x2]

    def ensure_rgb(x):
        if x.ndim == 2:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        if x.ndim == 3 and x.shape[2] == 1:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        return x

    roi1 = ensure_rgb(roi1)
    roi2 = ensure_rgb(roi2)

    g1 = cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY)

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(g1, g2, None)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u, v = pol2cart(mag, ang)
    o_s = computeStrain(u, v)

    H, W = u.shape

    final_roi = np.zeros((H, W, 3))
    final_roi[:, :, 0] = u
    final_roi[:, :, 1] = v
    final_roi[:, :, 2] = o_s

    final_resized = cv2.resize(final_roi, (42, 42))

    return final_resized


# ----------------------
# Recognition (onset–apex)
# ----------------------
def feature_extraction_recognition2(dataset_name, final_images, final_samples):
    final_videos_samples = [videos for subjects in final_samples for videos in subjects]
    print("Running Recognition...")
    start = time.time()

    dataset = []

    for vid in range(len(final_images)):

        ref_img = final_images[vid][0]
        rect = get_eyebrow_rect(ref_img)

        if rect is None:
            print("Eyebrow not detected – skipping", vid)
            continue

        # =============================================================
        #   CASME2 — فقط یک بازه دارد: [[onset, apex, offset]]
        # =============================================================
        if dataset_name == "CASME2":

            onset, apex, offset = final_samples[vid][0][0]

            img1 = final_images[vid][onset]
            img2 = final_images[vid][apex]

            final_img = preprocess_mediapipe(img1, img2, rect)
            dataset.append(final_img)

        # =============================================================
        #   CASME_sq و SAMMLV — چندین گروه و هر گروه چند بازه سه‌تایی
        # =============================================================
        elif dataset_name in ["CASME_sq", "SAMMLV"]:
            # Only onset and apex
            for sample in final_videos_samples[vid]:
                onset = sample[0]
                apex = sample[1]
                img1 = final_images[vid][onset]
                img2 = final_images[vid][apex]
                final_image = preprocess_mediapipe(img1, img2,rect)
                dataset.append(final_image)
            print('Video', vid, 'Done')

        # =============================================================
        #   SMIC — onset, apex, offset داریم اما apex نامشخص و باید پیدا شود
        # =============================================================
        elif "SMIC" in dataset_name:

            onset, apex_old, offset = final_samples[vid][0][0]

            img1 = final_images[vid][onset]

            max_d = 0
            best = None

            for k in range(offset - onset):
                img2 = final_images[vid][onset + k]
                processed = preprocess_mediapipe(img1, img2, rect)
                score = processed.sum()

                if score > max_d:
                    max_d = score
                    best = processed
                    final_samples[vid][0][0][1] = onset + k   # apex جدید

            dataset.append(best)

        print("Video", vid, "Done")

    print("All Recognition Done.")
    end = time.time()
    print("Time:", end - start)

    return dataset


# ----------------------
# Spotting (sliding window k)
# ----------------------
def feature_extraction_spotting2(dataset_name, final_images, k):

    
    print("Running Spotting...")
    start = time.time()

    dataset = []

    for vid in range(len(final_images)):

        print("\nVideo:", vid)

        first_frame = final_images[vid][0]
        rect = get_eyebrow_rect(first_frame)

        if rect is None:
            print("Eyebrow not found – skipping video.")
            continue

        OFF_video = []

        for i in range(final_images[vid].shape[0] - k):
            img1 = final_images[vid][i]
            img2 = final_images[vid][i + k]

            final_img = preprocess_mediapipe(img1, img2, rect)
            OFF_video.append(final_img)

        dataset.append(OFF_video)

        print("Video", vid, "Done.")

    print("\nAll Spotting Done.")
    end = time.time()
    print("Time:", end - start)

    return dataset
