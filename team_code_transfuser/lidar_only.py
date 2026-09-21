import torch
from torch import nn
import timm

class LidarOnlyBackbone(nn.Module):
    """
    LiDAR-only ablation backbone. No image branch, no image/lidar fusion.
    Used to measure how much a single camera or LiDAR modality contributes,
    as a counterpart to the (pre-existing) image-only latentTF backbone.
    image_architecture is accepted for a uniform constructor signature with
    the other backbones but is unused.
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input.
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=0):
        super().__init__()
        self.config = config

        if (config.use_point_pillars == True):
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if (self.config.use_target_point_image == True):
            in_channels += 1

        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)

        if (lidar_architecture.startswith('convnext')):
            self.norm_after_pool_lidar = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_lidar = nn.Sequential()

        # velocity embedding
        self.use_velocity = use_velocity
        if(use_velocity):
            self.vel_emb = nn.Linear(1, self.config.perception_output_features)

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)

        if(self.lidar_encoder._model.num_features != self.config.perception_output_features):
            self.reduce_channels_conv_lidar = nn.Conv2d(self.lidar_encoder._model.num_features, self.config.perception_output_features, (1, 1))
        else:
            self.reduce_channels_conv_lidar = nn.Sequential()

        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))

        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))

    def top_down(self, c5):

        p5 = self.relu(self.c5_conv(c5))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))

        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        '''
        LiDAR-only forward pass. `image` is accepted but ignored.
        Args:
            lidar (tensor): input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        # LiDAR branch
        output_features_lidar = self.lidar_encoder._model.forward_features(lidar)
        output_features_lidar = self.reduce_channels_conv_lidar(output_features_lidar)
        lidar_features_grid = output_features_lidar
        features = self.top_down(lidar_features_grid)

        lidar_features = torch.nn.AdaptiveAvgPool2d((1,1))(output_features_lidar)
        lidar_features = torch.flatten(lidar_features, 1)
        fused_features = self.norm_after_pool_lidar(lidar_features)

        if(self.use_velocity):
            velocity_embeddings = self.vel_emb(velocity) # (B, C)
            fused_features = torch.add(fused_features, velocity_embeddings)

        # No camera branch exists, so there is no meaningful camera-view feature
        # map for the (optional) semantic/depth aux heads. Callers must train
        # this backbone with multitask disabled (--no_semantic_loss 1, config.multitask = False).
        image_features_grid = lidar_features_grid

        return features, image_features_grid, fused_features


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False, in_chans=in_channels)
        self._model.fc = nn.Sequential()
        self._model.global_pool = nn.Sequential()
        self._model.classifier = nn.Sequential()
        self._model.head = nn.Sequential()
