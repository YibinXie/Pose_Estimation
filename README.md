# Pose estimation
## News

## Introduction
The code for this project is based of [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). What i am doing is to implement some models, then train and valid the models on dataset.
## Main results
### Results on MPII val
| Arch               | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|------|----------|-------|-------|------|------|-------|------|----------|
| pose_resnet_50     | 96.4 |     95.3 |  89.0 |  83.2 | 88.4 | 84.0 |  79.6 | 88.5 |     34.0 |
| pose_resnet_101    | 96.9 |     95.9 |  89.5 |  84.4 | 88.4 | 84.5 |  80.7 | 89.1 |     34.0 |
| pose_resnet_152    | 97.0 |     95.9 |  90.0 |  85.0 | 89.2 | 85.3 |  81.3 | 89.6 |     35.0 |
| pose_hrnet_w32     | 97.1 |     95.9 |  90.3 |  86.4 | 89.1 | 87.1 |  83.3 | 90.3 |     37.7 |

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| pose_resnet_50     |    256x192 | 34.0M   |    8.9 | 0.704 | 0.886 |  0.783 |  0.671 |  0.772 | 0.763 | 0.929 |  0.834 |  0.721 |  0.824 |
| pose_resnet_101    |    256x192 | 53.0M   |   12.4 | 0.714 | 0.893 |  0.793 |  0.681 |  0.781 | 0.771 | 0.934 |  0.840 |  0.730 |  0.832 |
| pose_resnet_152    |    256x192 | 68.6M   |   15.7 | 0.720 | 0.893 |  0.798 |  0.687 |  0.789 | 0.778 | 0.934 |  0.846 |  0.736 |  0.839 |
| pose_hrnet_w32     |    256x192 | 28.5M   |    7.1 | 0.744 | 0.905 |  0.819 |  0.708 |  0.810 | 0.798 | 0.942 |  0.865 |  0.757 |  0.858 |
| pose_hrnet_w48     |    256x192 | 63.6M   |   14.6 | 0.751 | 0.906 |  0.822 |  0.715 |  0.818 | 0.804 | 0.943 |  0.867 |  0.762 |  0.864 |
| pose_hourglass_2   |    256x192 | 24.7M   |   17.2 | 0.749 | 0.898 |  0.820 |  0.716 |  0.816 | 0.802 | 0.938 |  0.866 |  0.760 | 0.863 |

## Quick start
Please refer HRNet.
