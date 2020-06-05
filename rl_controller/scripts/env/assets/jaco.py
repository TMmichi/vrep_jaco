from pyrep.robots.arms.arm import Arm

class jaco(Arm):
    def __init__(self,count:int=0):
        super().__init__(count, 'jaco', num_joints=6)