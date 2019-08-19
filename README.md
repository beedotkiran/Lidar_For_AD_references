# Lidar Point clound processing for Autonomous Driving
A list of references on lidar point cloud processing for autonomous driving

## Clustering/Segmentation (road/ground extraction, plane extraction)
* Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications [[git](https://github.com/VincentCheungM/Run_based_segmentation)]
* Time-series LIDAR Data Superimposition for Autonomous Driving [[pdf](http://lab.cntl.kyutech.ac.jp/~nishida/paper/2016/ThBT3.3.pdf)]
* Fast segmentation of 3D point clouds for ground vehicles [[ieee](https://ieeexplore.ieee.org/document/5548059)]
* An Improved RANSAC for 3D Point Cloud Plane Segmentation Based on Normal Distribution Transformation Cells
* Segmentation of Dynamic Objects from Laser Data [[pdf](https://upcommons.upc.edu/bitstream/handle/2117/14119/1259-Segmentation-of-Dynamic-Objects-from-Laser-Data.pdf?sequence=1&isAllowed=y)]
* A Fast Ground Segmentation Method for 3D Point Cloud [[pdf](http://jips-k.org/file/down?pn=463)]
* Ground Estimation and Point Cloud Segmentation using SpatioTemporal Conditional Random Field [[pdf](https://hal.inria.fr/hal-01579095/document)]
* Real-Time Road Segmentation Using LiDAR Data Processing on an FPGA [[pdf](https://arxiv.org/pdf/1711.02757.pdf)]
* Efficient Online Segmentation for Sparse 3D Laser Scans [[pdf](http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16pfg.pdf)], [[git](https://github.com/PRBonn/depth_clustering)]
* CNN for Very Fast Ground Segmentation in Velodyne LiDAR Data [[pdf](https://arxiv.org/pdf/1709.02128.pdf)]
* A Comparative Study of Segmentation and Classification Methods for 3D Point Clouds [[pdf](http://publications.lib.chalmers.se/records/fulltext/238602/238602.pdf)]
* Fast Multi-pass 3D Point Segmentation Based on a Structured Mesh Graph for Ground Vehicles [pdf](https://ieeexplore.ieee.org/abstract/document/8500552) [video](https://www.youtube.com/watch?v=cwmcuRnWJfE)
* Circular Convolutional Neural Networks for Panoramic Images and Laser Data [pdf](https://www.tu-chemnitz.de/etit/proaut/publications/schubert19_IV.pdf)

## Pointcloud Density 
* DBSCAN algorithm : [wiki](https://en.wikipedia.org/wiki/DBSCAN), A density-based algorithm for discovering clusters in large spatial databases with noise (1996) [pdf[(https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220)
* Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection [pdf](https://bdpi.usp.br/bitstream/handle/BDPI/51005/2709770.pdf?sequence=1)
* STD: Sparse-to-Dense 3D Object Detector for Point Cloud [pdf](https://arxiv.org/abs/1907.10471)
* Fast semantic segmentation of 3d point clounds with strongly varying density [[pdf](https://www.ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/timo-jan-isprs2016.pdf)]
* The Perfect Match: 3D Point Cloud Matching with Smoothed Densities [[pdf](https://arxiv.org/abs/1811.06879)]

## Registration and Localization
* Point Clouds Registration with Probabilistic Data Association [[git](https://github.com/ethz-asl/robust_point_cloud_registration)]
* Robust LIDAR Localization using Multiresolution Gaussian Mixture Maps for Autonomous Driving [[pdf](https://pdfs.semanticscholar.org/7292/1fc6b181cf75790664e482963d982ec9ac48.pdf)], [[Thesis](https://pdfs.semanticscholar.org/a7ce/36bbdf85f1dba6cf16f47ad3799618511960.pdf)]
* Automatic Merging of Lidar Point-Clouds Using Data from Low-Cost GPS/IMU Systems [[pdf](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1081&context=ece_facpub)]
* 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration [[pdf](https://arxiv.org/abs/1807.09413)]
* Incremental Segment-Based Localization in 3D Point Clouds [[pdf](http://www.gilitschenski.org/igor/publications/201807-ral-incremental_segmatch/ral18-incremental_segmatch.pdf)]

## Feature Extraction
* Fast Feature Detection and Stochastic Parameter Estimation of Road Shape using Multiple LIDAR [[pdf](https://www.ri.cmu.edu/pub_files/2008/9/peterson_kevin_2008_1.pdf)]
* Finding Planes in LiDAR Point Clouds for Real-Time Registration [[pdf](http://ilab.usc.edu/publications/doc/Grant_etal13iros.pdf)]
* Online detection of planes in 2D lidar [[pdf](https://pdfs.semanticscholar.org/6857/b602dd702664c20febd41dc984451fd97bb3.pdf)]
* A Fast RANSAC–Based Registration Algorithm for Accurate Localization in Unknown Environments using LIDAR Measurements [[pdf](http://vision.ucla.edu/papers/fontanelliRS07.pdf)]
* Hierarchical Plane Extraction (HPE): An Efficient Method For Extraction Of Planes From Large Pointcloud Datasets [[pdf](https://pdfs.semanticscholar.org/8217/61a207088e6015de845cc3f9e556e1c94be1.pdf)]
* A Fast and Accurate Plane Detection Algorithm for Large Noisy Point Clouds Using Filtered Normals and Voxel Growing [[pdf](https://hal-mines-paristech.archives-ouvertes.fr/hal-01097361/document)]

## Object detection and Tracking
* Learning a Real-Time 3D Point Cloud Obstacle Discriminator via Bootstrapping [pdf](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.385.6290)
* Terrain-Adaptive Obstacle Detection [[pdf](https://pdfs.semanticscholar.org/92f6/26e75f940a49ee80eaf0344dc493f5d8b2ee.pdf)]
* 3D Object Detection from Roadside Data Using Laser Scanners [[pdf](http://www-video.eecs.berkeley.edu/papers/JYT/spie-paper.pdf)]
* 3D Multiobject Tracking for Autonomous Driving : Masters thesis A S Abdul Rahman
* Motion-based Detection and Tracking in 3D LiDAR Scans [[pdf](http://ais.informatik.uni-freiburg.de/publications/papers/dewan16icra.pdf)]
* Lidar-histogram for fast road and obstacle detection [[pdf](http://www.chenliang.me/blog/wp-content/uploads/2017/07/lidarhistogram.pdf)]
* End-to-end Learning of Multi-sensor 3D Tracking by Detection [pdf](https://arxiv.org/pdf/1806.11534.pdf)
* Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [pdf](https://arxiv.org/abs/1809.05590)
* Deep tracking in the wild: End-to-end tracking using recurrent neural networks [[pdf](http://www.robots.ox.ac.uk/~mobile/Papers/2017_IJRR_Dequaire.pdf)]
* Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [[pdf](https://arxiv.org/abs/1809.05590)], [[video](https://www.youtube.com/watch?v=2DzH9COLpkU)]
* Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets, Methods, and Challenges [[pdf](https://arxiv.org/pdf/1902.07830.pdf)]
* Low resolution lidar-based multi-object tracking for driving applications [pdf](https://upcommons.upc.edu/bitstream/handle/2117/113342/1924-Low-resolution-lidar-based-multi-object-tracking-for-driving-applications.pdf)

## Classification/Supervised Learning
* PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [[link](http://stanford.edu/~rqi/pointnet/), [link2](http://stanford.edu/~rqi/pointnet2/)]
* SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud [pdf](https://arxiv.org/pdf/1710.07368.pdf)
* Improving LiDAR Point Cloud Classification using Intensities and Multiple Echoes [[pdf](https://hal.archives-ouvertes.fr/hal-01182604/document)]
* DepthCN: Vehicle Detection Using 3D-LIDAR and ConvNet [[pdf](http://home.isr.uc.pt/~cpremebida/files_cp/DepthCN_preprint.pdf)]
* 3D Object Localisation with Convolutional Neural Networks [[Thesis](https://github.com/oscarmcnulty/gta-3d-dataset/blob/master/3D-object-localisation-with-cnns.pdf)]
* SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud [[pdf](https://arxiv.org/pdf/1809.08495.pdf)]
* PointSeg: Real-Time Semantic Segmentation Based on 3D LiDAR Point Cloud [[pdf](https://arxiv.org/pdf/1807.06288.pdf)]
* Fast LIDAR-based Road Detection Using Fully Convolutional Neural Networks [[pdf](https://arxiv.org/abs/1703.03613)]
* ChipNet: Real-Time LiDAR Processing for Drivable Region Segmentation on an FPGA [[pdf](https://arxiv.org/pdf/1808.03506.pdf)]

## Maps / Grids / HD Maps / Occupancy grids/ Prior Maps
* Hierarchies of Octrees for Efficient 3D Mapping [pdf](https://www.ais.uni-bonn.de/papers/IROS-2011_Wurm_Holz.pdf)
* Adaptive Resolution Grid Mapping using Nd-Tree [[ieee](https://ieeexplore.ieee.org/document/5980084)], [[pdf](https://www.researchgate.net/publication/224252536_Finding_the_adequate_resolution_for_grid_mapping_-_Cell_sizes_locally_adapting_on-the-fly)], [[video](https://www.youtube.com/watch?v=PYMlo8Wb6qE)]
* LIDAR-Data Accumulation Strategy To Generate High Definition Maps For Autonomous Vehicles [[link](https://ieeexplore.ieee.org/document/8170357/)]
* Long-term 3D map maintenance in dynamic environments [[video](https://www.youtube.com/watch?v=cMgLyLpnsoU)]
* Detection and Tracking of Moving Objects Using 2.5D Motion Grids [[pdf](http://a-asvadi.ir/wp-content/uploads/itsc15.pdf)]
* 3D Lidar-based Static and Moving Obstacle Detection in Driving Environments: an approach based on voxels and multi-region ground planes [[pdf](http://patternrecognition.cn/perception/negative2016a.pdf)]
* Spatio–Temporal Hilbert Maps for Continuous Occupancy Representation in Dynamic Environments [[pdf](https://papers.nips.cc/paper/6541-spatio-temporal-hilbert-maps-for-continuous-occupancy-representation-in-dynamic-environments.pdf)]
* Dynamic Occupancy Grid Prediction for Urban Autonomous Driving: A Deep Learning Approach with Fully Automatic Labeling [[pdf](https://arxiv.org/pdf/1705.08781.pdf)]
 * Fast 3-D Urban Object Detection on Streaming Point Clouds [[pdf](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/workshops/w15/Paper%202.pdf)]
* Mobile Laser Scanned Point-Clouds for Road Object Detection and Extraction: A Review [[pdf](https://www.mdpi.com/2072-4292/10/10/1531)]
* Efficient Continuous-time SLAM for 3D Lidar-based Online Mapping [[pdf](https://www.ais.uni-bonn.de/papers/ICRA_2018_Droeschel.pdf)]
* DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_DeLS-3D_Deep_Localization_CVPR_2018_paper.pdf)],[[video](https://www.youtube.com/watch?v=M6lhkzKFEhA)]
* Recurrent-OctoMap: Learning State-based Map Refinement for Long-Term Semantic Mapping with 3D-Lidar Data [[pdf](https://arxiv.org/pdf/1807.00925.pdf)]
* HDNET: Exploiting HD Maps for 3D Object Detection [[pdf](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf)]

## End-To-End Learning
* Monocular Fisheye Camera Depth Estimation Using Semi-supervised Sparse Velodyne Data [[pdf](https://arxiv.org/pdf/1803.06192.pdf)]
* Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)]

## Lidar Datasets and Simulators
* nuScenes : public large-scale dataset for autonomous driving [[dataset](https://www.nuscenes.org/overview)]
* Paris-Lille-3D: A Point Cloud Dataset for Urban Scene Segmentation and Classification [[pdf](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w40/Roynard_Paris-Lille-3D_A_Point_CVPR_2018_paper.pdf) [dataset](http://npm3d.fr/paris-lille-3d)]
* A LiDAR Point Cloud Generator: from a Virtual World to Autonomous Driving [pdf](https://arxiv.org/pdf/1804.00103.pdf)
* Ford Campus Vision and Lidar Data Set [pdf](http://robots.engin.umich.edu/uploads/SoftwareData/Ford/ijrr2011.pdf) [dataset](http://robots.engin.umich.edu/SoftwareData/Ford)
* Oxford RobotCar dataset [dataset](https://robotcar-dataset.robots.ox.ac.uk/) 1 Year, 1000km: The Oxford RobotCar Dataset [pdf](https://robotcar-dataset.robots.ox.ac.uk/images/robotcar_ijrr.pdf)
* Udacity based simulator [link](http://wangyangevan.weebly.com/lidar-simulation.html), [git](https://github.com/EvanWY/USelfDrivingSimulator)
* Tutorial on Gazebo to simulate raycasting from Velodyne lidar [[link](http://gazebosim.org/tutorials?tut=guided_i1)]
* Udacity Driving Dataset [[link](https://github.com/udacity/self-driving-car/tree/master/datasets)]
* Virtual KITTI [[link](http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds)]
* LiDAR-Video Driving Dataset: Learning Driving Policies Effectively [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_LiDAR-Video_Driving_Dataset_CVPR_2018_paper.pdf)]

## Movement and flow estimation in Pointclouds
* Deep Lidar CNN to Understand the Dynamics of Moving Vehicles [pdf](http://www.iri.upc.edu/files/scidoc/2018-Deep-Lidar-CNN-to-Understand-the-Dynamics-of-Moving-Vehicles.pdf)
* Learning motion field of LiDAR point cloud with convolutional networks [link](https://www.sciencedirect.com/science/article/abs/pii/S016786551930176X)
* Hallucinating Dense Optical Flow from Sparse Lidar for Autonomous Vehicles [pdf](https://arxiv.org/pdf/1808.10542.pdf) [video](https://www.youtube.com/watch?v=94vQUwCZLxQ)

