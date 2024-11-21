import numpy as np
import mujoco

class TransMatrix:
    def rotation_x(self, angle):
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    def rotation_y(self, angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    def rotation_z(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    
    def compute_obj_pos(self, object_pose_cam, palm_wrt_cam):
        # Compute the object's position with respect to the palm camera frame
        R_object_cam = object_pose_cam[:3, :3]
        R_palm_cam = palm_wrt_cam[:3, :3]
        R_cam_palm = R_palm_cam.T
        x_obj_cam = object_pose_cam[:3, 3]
        x_palm_cam = palm_wrt_cam[:3, 3]

        obj_pos_palm = np.dot(R_cam_palm, x_obj_cam - x_palm_cam)
        return obj_pos_palm

    def compute_contact_locations(self, object_pose_cam, palm_wrt_cam, bs):
        # Compute the contact positions for the index and thumb fingers
        obj_pos_palm = self.compute_obj_pos(object_pose_cam, palm_wrt_cam)

        R_object_cam = object_pose_cam[:3, :3]
        R_palm_cam = palm_wrt_cam[:3, :3]
        R_cam_palm = R_palm_cam.T

        R_object_palm = np.dot(R_object_cam, R_cam_palm)
        contactpos_1 = (obj_pos_palm.reshape(3, 1) + np.dot(R_object_palm, bs[0].reshape(3, 1))).flatten()
        contactpos_2 = (obj_pos_palm.reshape(3, 1) + np.dot(R_object_palm, bs[1].reshape(3, 1))).flatten()
        return contactpos_1, contactpos_2

class OnlyPosIK:
    def __init__(self,xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv)) 
        
        self.step_size = 0.5
        self.tol = 0.01
        self.alpha = 0.5
        self.init_q = [0.0, 0.0, 0.0, 0.0]  
    
    def check_joint_limits(self, q):
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Gradient Descent pseudocode implementation
    def calculate(self, goal, site_name):
        site_id=self.model.site(site_name).id

        self.data.qpos = self.init_q
        mujoco.mj_forward(self.model, self.data)

        current_pose = self.data.site(site_id).xpos
        error = np.subtract(goal, current_pose)

        max_iterations = 100000
        iteration = 0

        while (np.linalg.norm(error) >= self.tol) and iteration < max_iterations:
            #calculate jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr,site_id)

            #calculate gradient
            grad = self.alpha * self.jacp.T @ error

            #compute next step
            self.data.qpos += self.step_size * grad

            #check joint limits
            self.check_joint_limits(self.data.qpos)

            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 

            #calculate new error
            error = np.subtract(goal, self.data.site(site_id).xpos)

            iteration += 1

        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached.")
        
        result = self.data.qpos.copy()
        return result