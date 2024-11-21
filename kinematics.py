import numpy as np
import mujoco

class TransMatrix:
    def compute_obj_pos(self, object_pose_cam, palm_wrt_cam):
        # Compute the object's position with respect to the palm camera frame
        obj_pos = np.dot(np.linalg.inv(palm_wrt_cam), object_pose_cam)
        return obj_pos

    def compute_contact_locations(self, object_pose_cam, palm_wrt_cam, bs):
        # Compute the contact positions for the index and thumb fingers
        contactpos_1 = np.dot(np.linalg.inv(palm_wrt_cam), object_pose_cam) + bs[0]
        contactpos_2 = np.dot(np.linalg.inv(palm_wrt_cam), object_pose_cam) + bs[1]
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