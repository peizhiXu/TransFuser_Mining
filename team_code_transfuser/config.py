import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 1 # input timesteps
    # use different seq len for image and lidar
    img_seq_len = 1 
    lidar_seq_len = 1
    pred_len = 4 # future waypoints predicted
    scale = 1 # image pre-processing
    img_resolution = (160, 704) # image pre-processing in H, W
    img_width = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
    lidar_resolution_width  = 256 # Width of the LiDAR grid that the point cloud is voxelized into.
    lidar_resolution_height = 256 # Height of the LiDAR grid that the point cloud is voxelized into.
    pixels_per_meter = 8.0 # How many pixels make up 1 meter. 1 / pixels_per_meter = size of pixel in meters
    # Mounting positions for the Komatsu HD465-7E0 mining truck. Upstream used
    # [1.3, 0.0, 2.5] and [1.3, 0.0, 2.3] for the Lincoln MKZ. The truck is
    # 9.40 m long and 4.52 m tall, so the sedan mount sits inside its body.
    # These values are the ones team_code_autopilot/data_agent_mine.py collects
    # with; they are also what submission_agent.py mounts at deployment, so the
    # two must not drift apart. Checked on 0325_5: no self-occlusion, nearest
    # LiDAR return 9.6 m, no returns within 5 m.
    lidar_pos = [3.5, 0.0, 4.8] # x, y, z mounting position of the LiDAR
    lidar_rot = [0.0, 0.0, -90.0] # Roll Pitch Yaw of LiDAR in degree

    camera_pos = [3.5, 0.0, 4.8] #x, y, z mounting position of the camera
    # Closed-loop inference captures a wide 960x480/120-degree image from each
    # camera and crops its central 320x160 area. That crop has a 60-degree
    # horizontal FOV and therefore matches the lower-cost collection cameras.
    camera_width = 960 # Raw online camera width in pixels
    camera_height = 480 # Raw online camera height in pixels
    camera_fov = 120 # Raw online camera horizontal FOV in degrees
    camera_crop_width = 320 # Per-camera image passed to the panorama
    camera_crop_height = 160
    camera_crop_fov = 60 # Effective FOV after the central crop
    camera_rot_0 = [0.0, 0.0, 0.0] # Roll Pitch Yaw of camera 0 in degree
    camera_rot_1 = [0.0, 0.0, -60.0] # Roll Pitch Yaw of camera 1 in degree
    camera_rot_2 = [0.0, 0.0, 60.0] # Roll Pitch Yaw of camera 2 in degree

    bev_resolution_width  = 160 # Width resoultion the BEV loss is upsampled to. Double check if width and height are swapped if you want to make them non symmetric.
    bev_resolution_height = 160 # Height resoultion the BEV loss is upsampled to. Double check if width and height are swapped if you want to make them non symmetric.
    use_target_point_image = False
    gru_concat_target_point = True
    augment = True
    inv_augment_prob = 0.1 # Probablity that data augmentation is applied is 1.0 - inv_augment_prob
    aug_max_rotation = 20 # degree
    debug = False # If true the model in and outputs will be visualized and saved into Os variable Save_Path
    sync_batch_norm = False # If this is true we convert the batch norms, to synced bach norms.
    train_debug_save_freq = 50 # At which interval to save debug files to disk during training

    bb_confidence_threshold = 0.3 # Confidence of a bounding box that is needed for the detection to be accepted

    # Lidar discretization, configuration only used for Point Pillars
    use_point_pillars = False
    max_lidar_points = 40000
    min_x = -16
    max_x = 16
    min_y = -32
    max_y = 0
    num_input = 9
    num_features = [32, 32]

    backbone = 'transFuser'

    # CenterNet parameters
    num_dir_bins = 12
    fp16_enabled = False
    center_net_bias_init_with_prob = 0.1
    center_net_normal_init_std = 0.001
    top_k_center_keypoints = 100
    center_net_max_pooling_kernel = 3
    channel = 64

    bounding_box_divisor = 2.0 # The height and width of the bounding box value was changed by this factor during data collection. Fix that for future datasets and remove
    draw_brake_threshhold = 0.5 # If the brake value is higher than this threshhold, the bb will be drawn with the brake color during visualization

    #Waypoint GRU
    gru_hidden_size = 64

    num_class = 7
    classes = {
        0: [0, 0, 0],  # unlabeled
        1: [0, 0, 255],  # vehicle
        2: [128, 64, 128],  # road
        3: [255, 0, 0],  # red light
        4: [0, 255, 0],  # pedestrian
        5: [157, 234, 50],  # road line
        6: [255, 255, 255],  # sidewalk
    }
    #Color format BGR
    classes_list = [
        [0, 0, 0],  # unlabeled
        [255, 0, 0],  # vehicle
        [128, 64, 128],  # road
        [0, 0, 255],  # red light
        [0, 255, 0],  # pedestrian
        [50, 234, 157],  # road line
        [255, 255, 255],  # sidewalk
    ]
    converter = [
        0,  # unlabeled
        0,  # building
        0,  # fence
        0,  # other
        4,  # pedestrian
        0,  # pole
        5,  # road line
        2,  # road
        6,  # sidewalk
        0,  # vegetation
        1,  # vehicle
        0,  # wall
        0,  # traffic sign
        0,  # sky
        0,  # ground
        0,  # bridge
        0,  # rail track
        0,  # guard rail
        0,  # traffic light
        0,  # static
        0,  # dynamic
        0,  # water
        0,  # terrain
        3,  # red light
        3,  # yellow light
        0,  # green light
        0,  # stop sign
        5,  # stop line marking
    ]

    # Optimization
    lr = 1e-4 # learning rate
    multitask = True # whether to use segmentation + depth losses
    ls_seg   = 1.0
    ls_depth = 10.0

    # Conv Encoder
    img_vert_anchors = 5
    img_horz_anchors = 20 + 2
    lidar_vert_anchors = 8
    lidar_horz_anchors = 8
    
    img_anchors = img_vert_anchors * img_horz_anchors
    lidar_anchors = lidar_vert_anchors * lidar_horz_anchors

    detailed_losses = ['loss_wp', 'loss_bev', 'loss_depth', 'loss_semantic', 'loss_center_heatmap', 'loss_wh',
                       'loss_offset', 'loss_yaw_class', 'loss_yaw_res', 'loss_velocity', 'loss_brake']
    detailed_losses_weights = [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

    perception_output_features = 512 # Number of features outputted by the perception branch.
    bev_features_chanels = 64 # Number of channels for the BEV feature pyramid
    bev_upsample_factor = 2

    deconv_channel_num_1 = 128 # Number of channels at the first deconvolution layer
    deconv_channel_num_2 = 64 # Number of channels at the second deconvolution layer
    deconv_channel_num_3 = 32 # Number of channels at the third deconvolution layer

    deconv_scale_factor_1 = 8 # Scale factor, of how much the grid size will be interpolated after the first layer
    deconv_scale_factor_2 = 4 # Scale factor, of how much the grid size will be interpolated after the second layer

    gps_buffer_max_len = 100 # Number of past gps measurements that we track.
    carla_frame_rate = 1.0 / 20.0 # CARLA frame rate in milliseconds
    carla_fps = 20 # Simulator Frames per second
    iou_treshold_nms = 0.2  # Iou threshold used for Non Maximum suppression on the Bounding Box predictions for the ensembles
    steer_damping = 0.5 # Damping factor by which the steering will be multiplied when braking
    route_planner_min_distance = 7.5
    route_planner_max_distance = 50.0
    action_repeat = 2 # Number of times we repeat the networks action. It's 2 because the LiDAR operates at half the frame rate of the simulation
    # Recovery timing is expressed in simulator frames and converted to the
    # 10 Hz inference loop.  The upstream agent waited 55 s, crept for only
    # 1.5 s and could never retry after a failed attempt.  A loaded HD465 needs
    # a longer launch window, especially on the 10--16% grades in this set.
    stuck_threshold = 200 // action_repeat       # 10 s before first attempt
    creep_duration = 60 // action_repeat         # 3 s per attempt
    stuck_retry_delay = 100 // action_repeat     # 5 s between attempts
    max_recovery_attempts = 3
    stuck_speed_threshold = 0.1
    stuck_release_speed = 0.5

    # HD465 dimensions and the LiDAR-frame region checked while the stuck
    # recovery controller tries to creep forward. Coordinates follow the
    # TransFuser LiDAR convention: x is lateral and negative y is forward.
    # Ignore the road plane.  With a 4.8 m LiDAR, a 16% uphill can rise to
    # roughly z=-3.5 m at 8 m range; the former -4.6 m lower bound classified
    # that road surface as an obstacle and cancelled every recovery attempt.
    safety_box_z_min = -3.0
    safety_box_z_max = 0.5

    safety_box_y_min = -8.0
    safety_box_y_max = -1.0

    safety_box_x_min = -2.8
    safety_box_x_max = 2.8

    ego_extent_x = 4.698465 # Half the HD465 length in x direction
    ego_extent_y = 2.682265 # Half the HD465 width in y direction
    ego_extent_z = 2.261520 # Half the HD465 height in z direction

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0 # Mean of the normal distribution with which the linear layers in the GPT are initialized
    gpt_linear_layer_init_std  = 0.02 # Std  of the normal distribution with which the linear layers in the GPT are initialized
    gpt_layer_norm_init_weight = 1.0 # Initial weight of the layer norms in the gpt.

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 20 # buffer size

    # Speed PID and clip values below match team_code_autopilot/autopilot_mine.py's
    # override for the HD465 (heavier, slower-responding than the Lincoln MKZ
    # these were originally tuned for). turn_KP/KI/KD above are left at the
    # upstream values because autopilot_mine.py never overrides the steering
    # controller. Deployment (submission_agent.py) builds its PID controllers
    # straight from this config (model.py:608-609), so a mismatch here would
    # only surface at closed-loop time, not during training.
    speed_KP = 0.28
    speed_KI = 0.12
    speed_KD = 0.04
    speed_n = 20 # buffer size

    default_speed = 4.0 # Speed used when creeping

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 1.5 # maximum change in speed input to logitudinal controller
    clip_throttle = 1.0 # Maximum throttle allowed by the controller

    # Longitudinal feed-forward and service-brake values mirror the mining
    # expert.  They are deployment controller parameters, not learned weights.
    cruise_feed_forward = 0.12
    uphill_feed_forward_gain = 5.8
    service_brake_gain = 0.22
    service_brake_deadband = 0.25
    max_service_brake = 0.65
    downhill_brake_gain = 3.2
    max_downhill_brake = 0.60

    def __init__(self, root_dir='', setting='all', **kwargs):
        self.root_dir = root_dir
        if (setting == 'all'): # All towns used for training no validation data
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = [self.train_towns[0]]
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.val_data.append(os.path.join(self.root_dir, town, file))

        elif (setting == '02_05_withheld'): #Town02 and 05 withheld during training
            print("Skip Town02 and Town05")
            self.train_towns = os.listdir(self.root_dir) #Scenario Folders
            self.val_towns = self.train_towns # Town 02 and 05 get selected automatically below
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if ((file.find('Town02') != -1) or (file.find('Town05') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Train Folder: ", file)
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if ((file.find('Town02') == -1) and (file.find('Town05') == -1)): # Only use Town 02 and 05 for validation
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Val Folder: ", file)
                        self.val_data.append(os.path.join(self.root_dir, town, file))
        elif (setting == 'mining'):
            # Explicit route-level split: root_dir/train and root_dir/val.
            train_root = os.path.join(self.root_dir, 'train')
            val_root = os.path.join(self.root_dir, 'val')
            if not os.path.isdir(train_root) or not os.path.isdir(val_root):
                raise ValueError(
                    'Mining setting expects train and val directories under: %s'
                    % self.root_dir)
            self.train_towns = ['train']
            self.val_towns = ['val']
            self.train_data = [train_root]
            self.val_data = [val_root]
        elif (setting == 'eval'): #No training data needed during evaluation.
            pass
        else:
            print("Error: Selected setting: ", setting, " does not exist.")

        for k,v in kwargs.items():
            setattr(self, k, v)
