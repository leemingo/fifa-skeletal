import numpy as np
import cv2
import os
from pathlib import Path
import torch
import tqdm

# Importing necessary modules from aitviewer
import sys
sys.path.append("../aitviewer")
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node

# Importing local modules
from lib.pitch import SoccerPitch
from lib.utils import project_points

PITCH = SoccerPitch()
SOCCER_FIELD_LINES = PITCH.lines


def create_billboard(camera, img_folder, distance=200, draw_fn=None):
    """Create a billboard from a sequence of images."""
    img_paths = sorted(img_folder.glob("*.jpg"))
    H, W = camera.rows, camera.cols
    pc = Billboard.from_camera_and_distance(
        camera, distance, W, H, textures=[str(path) for path in img_paths], image_process_fn=draw_fn
    )
    return pc


def convert_video_to_images(video_path, output_folder):
    """Convert video to images."""
    output_folder.mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    with tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Converting video to images:") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fp = output_folder / f"{frame_id:06d}.jpg"
            if not fp.exists():
                cv2.imwrite(str(fp), frame)
            frame_id += 1
            pbar.update(1)
    cap.release()


def create_smpl_sequences(params, names=None, colors=None, post_fk_func=None):
    """Create SMPL sequences for animation."""
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
    num_subjects = len(params["global_orient"])

    names = names if names is not None else [f"SMPL_{i}" for i in range(num_subjects)]
    colors = colors if colors is not None else [(0.5, 0.5, 0.5, 1) for _ in range(num_subjects)]

    smpl_seqs = []
    for i in range(num_subjects):
        smpl_seq = SMPLSequence(
            poses_body=params["body_pose"][i],
            smpl_layer=smpl_layer,
            poses_root=params["global_orient"][i],
            betas=params["betas"][i],
            trans=params["transl"][i],
            is_rigged=False,
            post_fk_func=post_fk_func,
            name=names[i],
            color=colors[i],
        )
        smpl_seq.mesh_seq.compute_vertex_and_face_normals.cache_clear()
        smpl_seqs.append(smpl_seq)
    return smpl_seqs


def draw_field(frame, calib, img_shape):
    for line in SOCCER_FIELD_LINES:
        line = project_points(line, **calib, img_shape=img_shape)
        line = line.astype(np.int32)
        cv2.polylines(frame, [line], False, (0, 0, 255), 5)
    return frame


def make_draw_func(camera=None):
    def _draw_func(img, current_frame_id):
        if camera:
            current_frame_id = min(current_frame_id, len(camera["K"]) - 1)
            img_shape = img.shape
            calib = {
                "R": camera["R"][current_frame_id],
                "t": camera["t"][current_frame_id],
                "k": camera["k"][current_frame_id],
                "f": camera["K"][current_frame_id][0, 0],
                "principal_points": camera["K"][current_frame_id][:2, 2],
            }
            draw_field(img, calib, img_shape)
        return img

    return _draw_func


def make_post_fk_func(camera_params):
    R_all = torch.from_numpy(camera_params["R"]).float()
    t_all = torch.from_numpy(camera_params["t"]).float()
    k_all = torch.from_numpy(camera_params["k"]).float()

    def _post_fk_func(self, vertices, joints, current_frame_only):
        # apply rotation and translation
        nonlocal R_all, t_all, k_all
        R = R_all.to(vertices.device)
        t = t_all.to(vertices.device)
        k = k_all.to(vertices.device)
        if current_frame_only:
            R = R[self.current_frame_id][None]
            t = t[self.current_frame_id][None]
            k = k[self.current_frame_id][None]

        vertices_cam = (R[:, None] @ vertices[..., None]).squeeze(-1)
        vertices_cam += t[:, None]
        vertices_normalized = vertices_cam[..., :2] / vertices_cam[..., 2:]
        r = vertices_normalized.square().sum(-1, keepdims=True)
        # r.clamp_(0, 1)
        k1 = k[:, 0:1]
        k2 = k[:, 1:2]

        scale = 1 + k1[:, None] * r + k2[..., None] * r.square()
        vertices_cam[..., :2] *= scale

        # transform back to world coordinates
        vertices_cam -= t[:, None]
        vertices_cam = (R[:, None].transpose(-1, -2) @ vertices_cam[..., None]).squeeze(-1)
        vertices = vertices_cam
        return vertices, joints

    return _post_fk_func

def run_viewer(camera_params, img_folder, smpl_seqs):
    # Setup viewer and load data
    viewer = Viewer(size=(1920, 1080))
    # Setup camera and billboard
    camera = OpenCVCamera(camera_params["K"], camera_params["Rt"], 1920, 1080, viewer=viewer, name="Overlay")
    billboard = create_billboard(camera, img_folder, 200, make_draw_func(camera_params))
    viewer.scene.add(billboard)
    viewer.scene.add(camera)
    # Add SMPL sequences to the scene
    smpl_seq_node = Node(name="SMPL", n_frames=len(camera_params["K"]), is_selectable=False)
    for seq in smpl_seqs:
        smpl_seq_node.add(seq)
    viewer.scene.add(smpl_seq_node)
    # Configure lighting
    light = viewer.scene.lights[0]
    light.shadow_enabled = True
    light.azimuth = 270
    light.elevation = 0
    light.shadow_map_size = 64
    light.shadow_map_near = 0.01
    light.shadow_map_far = 50
    viewer.scene.lights[1].shadow_enabled = False

    # Finalize setup and run viewer
    viewer.scene.floor.enabled = False
    viewer.set_temp_camera(camera)
    viewer.run()
    return True


def save_image(camera_params, img_folder, smpl_seqs, frame_num):
    image_path = img_folder / f"{frame_num:06d}.jpg"

        # Read image in BGR format
    img = cv2.imread(image_path)

    # Camera parameters
    R = camera_params["R"][frame_num]
    t = camera_params["t"][frame_num]
    k = camera_params["k"][frame_num]
    f = camera_params["K"][frame_num][0, 0]
    principal_points = camera_params["K"][frame_num][:2, 2]
    img_shape = img.shape

    # Draw soccer field lines
    for line in SOCCER_FIELD_LINES:
        line = project_points(line, R=R, t=t, f=f, principal_points=principal_points, k=k, img_shape=img_shape)
        line = line.astype(np.int32)
        cv2.polylines(img, [line], isClosed=False, color=(0, 0, 255), thickness=2)  # Red lines

    # Visualize SMPL joints
    for smpl_seq in smpl_seqs:
        joints = smpl_seq.joints[frame_num]  # frame's joint coordinates
        if np.isnan(joints).all():  # Skip if all joints are NaN
            continue

        # Project joints to 2D
        joints_2d = project_points(joints, R=R, t=t, f=f, principal_points=principal_points, k=k)
        
        # Draw joint positions
        for joint in joints_2d:
            x, y = joint.astype(int)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Green circles

        # Draw joint connections (skeleton)
        skeleton = smpl_seq.smpl_layer.skeletons()["body"]  # Body joint connections
        for parent_idx, child_idx in skeleton.T:
            if parent_idx == -1:  # Skip root joint
                continue

            parent_joint = joints_2d[parent_idx]
            child_joint = joints_2d[child_idx]

            # Skip if NaN in joint
            if np.isnan(parent_joint).any() or np.isnan(child_joint).any():
                continue

            # Draw line between parent and child joints
            parent_x, parent_y = parent_joint.astype(int)
            child_x, child_y = child_joint.astype(int)
            cv2.line(img, (parent_x, parent_y), (child_x, child_y), (255, 0, 0), 2)  # Blue line

    # Save the output image
    cv2.imwrite(f"visualize_{clip_name}_{frame_num}.jpg", img)
    print(f"The output image has been saved to visualize_{clip_name}_{frame_num}.jpg.")    

def save_video(video_path, camera_params, img_folder, smpl_seqs):
    output_video_path = img_folder.with_name(f"{clip_name}_video.mp4")
    output_image_path = img_folder.with_name(f"{clip_name}_overlay")
    # Create a directory to save the output images
    os.makedirs(output_image_path, exist_ok=True)
    # Create video capture object
    cap = cv2.VideoCapture(video_path)

    # Get video frame size and FPS information
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer object (save to output_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    frame_num = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        R = camera_params["R"][frame_num]
        t = camera_params["t"][frame_num]
        k = camera_params["k"][frame_num]
        f = camera_params["K"][frame_num][0, 0]
        principal_points = camera_params["K"][frame_num][:2, 2]
        img_shape = img.shape

        # Draw soccer field lines
        for line in SOCCER_FIELD_LINES:
            line = project_points(line, R=R, t=t, f=f, principal_points=principal_points, k=k, img_shape=img.shape)
            line = line.astype(np.int32)
            cv2.polylines(img, [line], isClosed=False, color=(0, 0, 255), thickness=2)  # Red lines

        # Visualize SMPL joints on the image (e.g., project SMPL joint coordinates to 2D)
        for smpl_seq in smpl_seqs:
            joints = smpl_seq.joints[frame_num]  # Joint coordinates of the first frame (e.g., body, hands)
            if np.isnan(joints).all():  # If joints contain NaN values
                continue  # Skip frames with NaN values

            # Project joint 3D coordinates to 2D image coordinates
            joints_2d = project_points(joints, R=R, t=t, f=f, principal_points=principal_points, k=k)

            # Visualize joint positions (draw a circle for each joint)
            for joint in joints_2d:
                x, y = joint.astype(int)
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Green circles for joints

            # Draw skeleton joints connections (parent-child relationships)
            skeleton = smpl_seq.smpl_layer.skeletons()["body"]  # SMPL body joint connections
            for parent_idx, child_idx in skeleton.T:  # Parent-child relationships
                if parent_idx == -1:  # Skip root joint as it has no parent
                    continue

                parent_joint = joints_2d[parent_idx]
                child_joint = joints_2d[child_idx]

                # Skip joints that contain NaN values
                if np.isnan(parent_joint).any() or np.isnan(child_joint).any():
                    continue

                # Check if the coordinates are within the image boundaries
                if not (0 <= parent_joint[0] < frame_width and 0 <= parent_joint[1] < frame_height) or \
                not (0 <= child_joint[0] < frame_width and 0 <= child_joint[1] < frame_height):
                    continue  # Skip joints outside the image range

                parent_x, parent_y = parent_joint.astype(int)
                child_x, child_y = child_joint.astype(int)

                # Draw line connecting parent and child joints
                cv2.line(img, (parent_x, parent_y), (child_x, child_y), (255, 0, 0), 2)  # Blue line

        frame_num += 1
        # Save the processed frame to the output video
        out.write(img)
        # Save the frame image
        cv2.imwrite(f"{output_image_path}/{frame_num:06d}.jpg", img)

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Print message
    print(f"The output video has been saved to {output_video_path}.")    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="ARG_FRA_182345")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--use_viewer", action="store_true", help="Use aitviewer for visualization")
    parser.add_argument("--frame_num", type=int, default=0, help="Frame number to process (0 for all)")
    parser.add_argument("--output_type", type=str, choices=["images", "video", "both"], default="both", help="Choose output type: 'images', 'video', or 'both'")
    args = parser.parse_args()

    # Define constants
    C.z_up = True
    C.smplx_models = f"{args.data_path}/models"
    
    # Load data
    clip_name = args.sequence
    data_dir = Path(args.data_path)
    
    smpl_param_path = data_dir / f"poses/{clip_name}.npz"
    video_path = data_dir / f"videos_train/{clip_name}.mp4"
    calibration_path = data_dir / f"cameras_train/{clip_name}.npz"
    if not smpl_param_path.exists():
        raise FileNotFoundError(f"SMPL parameters not found at {smpl_param_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration parameters not found at {calibration_path}")

    camera_params = dict(np.load(calibration_path))

    # Load SMPL parameters
    smpl_params = dict(np.load(smpl_param_path))
    colors = np.random.rand(len(smpl_params["betas"]), 4)
    colors[:, 3] = 1
    post_fk_func = make_post_fk_func(camera_params)
    smpl_seqs = create_smpl_sequences(smpl_params, colors=colors, post_fk_func=post_fk_func)

    img_folder = Path(f"{args.data_path}/outputs/{clip_name}")
    ## create a tmp folder and convert video to images
    if os.path.exists(img_folder) == False:
        convert_video_to_images(video_path, img_folder)
    
    if args.use_viewer:
        # Use aitviewer for visualization
        run_viewer(camera_params, img_folder, smpl_seqs) 
    else:
        # Save image and video using cv2
        if args.output_type in ["images", "both"]:
            # Save image with overlays
            save_image(camera_params, img_folder, smpl_seqs, args.frame_num)
        
        if args.output_type in ["video", "both"]:
            # Save video with overlays
            save_video(video_path, camera_params, img_folder, smpl_seqs)
        
