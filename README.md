# itr-leaphand-graspobject
This repository contains codes required for the guided assignment for ME 639 Introduction to Robotics course at IIT Gandhinagar. The task was to enable a LEAP Hand to hold on to an object based on the 6D live pose of the object.

**Required transformations:**  
![Transformations](Transformations.jpg "These are the required transformations")
![MeshLabss](MeshLabss.jpg "This is a screenshot from the MeshLab software showing palm_lower.stl")

**'Compute Geometric Measure' result from mesh lab:**  
Opened mesh /home/iitgn-robotics/Saniya/redundancy-leap/leap-mujoco/model/leap hand/palm_lower.stl in 105 msec  
All files opened in 105 msec  
Recentering on point [-0.019383 -0.042410 -0.000013] [823,489]  
Selected new Mesh 0  
Enabled Decorate mode Show Axis  
Started Mode Measuring Tool  
Mesh Bounding Box Size 0.103000 0.125729 0.046000  
Mesh Bounding Box Diag 0.168917   
Mesh Bounding Box min -0.080000 -0.125729 0.000000  
Mesh Bounding Box max 0.023000 0.000000 0.046000  
Mesh Surface Area is 0.099924  
Mesh Total Len of 130399 Edges is 224.693970 Avg Len 0.001723  
Mesh Total Len of 130399 Edges is 224.693970 Avg Len 0.001723 (including faux edges))  
Thin shell (faces) barycenter: -0.025112 -0.062744 0.023246  
Vertices barycenter -0.003146 -0.056586 0.019961  
Mesh is not 'watertight', no information on volume, barycenter and inertia tensor.  
Principal axes are :  
| -0.122957 -0.978845 0.163534 |  
| 0.015599 -0.166670 -0.985889 |  
| 0.992289 -0.118671 0.035762 |  
Applied filter Compute Geometric Measures in 135 msec  

**Given:**  
main.py is the library for the Leap Hand API. Please refer to leaphand/LEAP_Hand_API for installation if not already installed.
index_finger.xml and thumb.xml are the original xml files for index and thumb finger. Create new modified xml files with the necessary transformations.  
test_receiver_4oct.py is the receiver code for tcp/ip communication. It receives a 4x4 transformation matrix of the .obj provided in foundationpose pipeline.
test_invkin_11oct.py performs inverse kinematics for index and thumb individually and uses the dynamixel sdk to give commands to the leaphand. 
