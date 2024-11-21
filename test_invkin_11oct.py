import pickle
import numpy as np

#write your own library with TransMatrix and OnlyPosIK classes as required 
from main_redundancy import TransMatrix, OnlyPosIK

from main import LeapNode
import time

transmatrix = TransMatrix()
leap=LeapNode()

index_path='/home/abhinab/LEAP_Hand_API/python/index_fingerxml'
thumb_path='/home/abhinab/LEAP_Hand_API/python/thumb.xml'

pos_ik_index=OnlyPosIK(index_path)
pos_ik_thumb=OnlyPosIK(thumb_path)

T_indexbase_palm=np.array([[0.        , 0.        , 1.        , 0.01309895],
    [1.        , 0.        , 0.        , -0.0027],
    [0.        , 1.        , 0.        , 0.016012  ],
    [0.        , 0.        , 0.        , 1.        ]])
T_palm_indexbase = np.linalg.inv(T_indexbase_palm)

T_thumbbase_palm=np.array([[0.        , 0.        , 1.        , -0.0493],
    [0.        , 1.        , 0.        , -0.0027],
    [-1.        , 0.        , 0.        , 0.0131  ],
    [0.        , 0.        , 0.        , 1.        ]])
# n = 2
# palm_wrt_cam =  np.array(
# [[-0.04926246, 0.96736079, 0.24856618, 0.07521848],
#  [-0.82615805, 0.10038926, -0.55442315, -0.00205819],
#  [-0.56128109, -0.23266779, 0.79424787, 0.45737721],
#  [ 0., 0., 0., 1.0]]
# )

palm_wrt_cam =  np.array([
    [-0.01848555, 0.97723478, 0.21134862, 0.03405158],
    [-0.87437814, 0.08671534, -0.47743279, 0.04338642],
    [-0.48489177, -0.19362427, 0.85287136, 0.4968603],
    [0., 0., 0., 1.]
])


b1 = np.array([[0],[-0.027],[0]])
b2 = np.array([[0],[0.027],[0]])
bs = [b1, b2]


def f(array):
    object_pose_cam=array
    obj_pos=transmatrix.compute_obj_pos(object_pose_cam, palm_wrt_cam) #object position with respect to palm camera frame
    print('obj_pos',obj_pos)

    contactpos_1,contactpos_2=transmatrix.compute_contact_locations(object_pose_cam, palm_wrt_cam,bs)
    print('contact1',contactpos_1)
    print('contact2',contactpos_2)

    
    qs1=pos_ik_index.calculate(contactpos_1,'contact_index')
    qs2=pos_ik_thumb.calculate(contactpos_2,'contact_thumb')
    qs1_real=qs1
    qs2_real=qs2
    qs1_real[0],qs1_real[1]=qs1_real[1],qs1_real[0]
    qs=np.concatenate([qs1_real,np.zeros(8),qs2_real])

    while True:
        leap.set_allegro(qs)

def load_and_compute(filename):
    """
    # Load the array from a pickle file and compute f(array).
    # """
    try:
        with open(filename, 'rb') as file:  
            print(f"Type of file: {type(file)}")  
            print("Attempting to load array from file...")

            array = pickle.load(file) 
            
            result = f(np.array(array))
            

    except FileNotFoundError:
        print(f"File '{filename}' not found. Make sure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    filename = '/tmp/received_array.pkl'
    load_and_compute(filename)
    
