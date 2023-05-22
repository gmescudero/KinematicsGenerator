from cmath import pi
import numpy as np
import sympy
from dataclasses import dataclass
from enum import Enum

def printM(name:str, M):
    try: 
        M = sympy.nsimplify(M,tolerance=1e-12,rational=True).evalf()
    except Exception:
        pass

    print(f"{name}\t---------------------------------------\n"+
          f"> pos:\t{float(M[0,3]):.2f}\t{float(M[1,3]):.2f}\t{float(M[2,3]):.2f} \n"+
          f"> ori:\t{float(M[0,0]):.2f}\t{float(M[0,1]):.2f}\t{float(M[0,2]):.2f} \n"+
          f"      \t{float(M[1,0]):.2f}\t{float(M[1,1]):.2f}\t{float(M[1,2]):.2f} \n"+
          f"      \t{float(M[2,0]):.2f}\t{float(M[2,1]):.2f}\t{float(M[2,2]):.2f}")
    

class OrientationType(Enum):
    NONE = -1
    MATRIX = 0
    EULER = 1
    QUATERNION = 2

class Rotations:
    def __init__(self):
        self.alpha = sympy.Symbol('alpha')
        self.beta  = sympy.Symbol('beta')
        self.gamma = sympy.Symbol('gamma')
        self.theta = sympy.Symbol('theta')
        # Nominal rotations
        self.zRotM = sympy.rot_axis3(self.theta).transpose()
        self.yRotM = sympy.rot_axis2(self.theta).transpose()
        self.xRotM = sympy.rot_axis1(self.theta).transpose()
        # Rodriges formula
        self.p = sympy.Matrix([sympy.Symbol('p_x'),sympy.Symbol('p_y'),sympy.Symbol('p_z'),])
        self.k = sympy.Matrix([sympy.Symbol('k_x'),sympy.Symbol('k_y'),sympy.Symbol('k_z'),])
        skewK = sympy.Matrix([
            [ 0,         self.k[2], self.k[1]],
            [ self.k[2], 0,        -self.k[0]],
            [-self.k[1], self.k[0], 0]
        ])
        self.rodriges = sympy.eye(3)*sympy.cos(self.alpha) + (1-sympy.cos(self.alpha))*(self.k*self.k.T) + sympy.sin(self.alpha)*skewK
        self.screwMatrix = sympy.Matrix([
            [self.rodriges,     (sympy.eye(3)-self.rodriges)*self.p],
            [sympy.zeros(1,3),  sympy.eye(1)                       ]
        ])
    
    def eulerToMatrixSequenceSym(self,sequence:str, a=sympy.Symbol('a'),b=sympy.Symbol('b'),c=sympy.Symbol('c')):
        if len(sequence) != 3: return None
        rotSymbols = [a,b,c]
        rotM = sympy.eye(3)
        for axis,symbol in zip(list(sequence),rotSymbols):
            if   axis == "x":
                rotM = rotM*self.xRotM.subs(self.theta,symbol)
            elif axis == "y":
                rotM = rotM*self.yRotM.subs(self.theta,symbol)
            elif axis == "z":
                rotM = rotM*self.zRotM.subs(self.theta,symbol)
        return rotM
    
    def matrixToEulerSequenceSym(self,sequence:str, rotM=sympy.MatrixSymbol('R',3,3)):
        euler = [0,0,0]
        if sequence == "xyz":
            euler[0] = sympy.atan2(-rotM[1,2], rotM[2,2])
            euler[1] = sympy.asin(  rotM[0,2])
            euler[2] = sympy.atan2(-rotM[0,1], rotM[0,0])
        elif sequence == "xyx":
            euler[0] = sympy.atan2( rotM[1,0],-rotM[2,0])
            euler[1] = sympy.acos(  rotM[0,0])
            euler[2] = sympy.atan2( rotM[0,1], rotM[0,2])
        elif sequence == "zyz":
            euler[0] = sympy.atan2( rotM[1,2], rotM[0,2])
            euler[1] = sympy.acos(  rotM[2,2])
            euler[2] = sympy.atan2( rotM[2,1],-rotM[2,0])
        else:
            print(f"WARNING: Euler sequence \"{sequence}\" not supported")
            return None
        return sympy.Matrix(euler)
    
    def eulerToMatrixSequence(self,sequence:str, a,b,c):
        sym = self.eulerToMatrixSequenceSym(sequence,a,b,c)
        return np.array(sym.evalf())
    
    def matrixToEulerSequence(self, sequence:str, rotM):
        sym = self.matrixToEulerSequenceSym(sequence,rotM)
        return np.array(sym.evalf())

    def zyxToMatrixSym(self,z,y,x):
        return self.eulerToMatrixSequenceSym("zyx",z,y,x)
    
    def xyzToMatrixSym(self,x,y,z):
        return self.eulerToMatrixSequenceSym("xyz",x,y,z)

    def matrixToEulerXYZSym(self, matrix):
        if matrix.shape[0] == 4:
            pos = matrix[0:3,3]
        R = matrix[0:3,0:3]
        sy = sympy.sqrt(R[0,0]**2 +  R[1,0]**2)
        x = sympy.atan2(R[2,1] , R[2,2])
        y = sympy.atan2(-R[2,0], sy)
        z = sympy.atan2(R[1,0], R[0,0])
        if matrix.shape[0] == 4:
            return sympy.Matrix([[pos[0]],[pos[1]],[pos[2]],[x],[y],[z]])
        return sympy.Matrix([[x],[y],[z]])
        
    def matrixToEulerXYZ(self, matrix):
        if matrix.shape[0] == 4:
            pos = matrix[0:3,3]
        R = matrix[0:3,0:3]
        sy = np.sqrt(R[0,0]**2 +  R[1,0]**2)
        if  sy > 1e-6 :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            # print("Singular")
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        if matrix.shape[0] == 4:
            return np.transpose(np.array([[pos[0], pos[1], pos[2], x, y, z]]))
        return np.transpose(np.array([x, y, z]))

class JointType(Enum):
    """
    This Enums the joint types supported 
    """
    ROTATIONAL = 0
    PRISMATIC = 1

@dataclass
class Joint:
    """
    This classes define a single joint in symbolic representation and stores its type 
    """
    symbol: sympy.Symbol
    type: JointType
    name: str = None
    upper_limit: float = 2*pi
    lower_limit: float =-2*pi

    def __post_init__(self):
        if self.name is None:
            self.name = str(self.symbol)

class Denavit:
    """
    Denavit Hartenberg table row to matrix
    """
    def __init__(self):
        self.theta = sympy.Symbol('theta')
        self.alfa = sympy.Symbol('alpha') 
        self.d = sympy.Symbol('d')
        self.a = sympy.Symbol('a')

        cosT = sympy.cos(self.theta)
        sinT = sympy.sin(self.theta)
        cosA = sympy.cos(self.alfa)
        sinA = sympy.sin(self.alfa)
        
        self.denavit = sympy.Matrix([
            [cosT,  -sinT*cosA, sinT*sinA,  self.a*cosT ],
            [sinT,  cosT*cosA,  -cosT*sinA, self.a*sinT ],
            [0,     sinA,       cosA,       self.d      ],
            [0,     0,          0,          1           ]
        ])
        self.position = sympy.Matrix([self.a*cosT, self.a*sinT, self.d])

        cosT2 = sympy.cos(self.theta/2)
        sinT2 = sympy.sin(self.theta/2)
        cosA2 = sympy.cos(self.alfa/2)
        sinA2 = sympy.sin(self.alfa/2)
        self.quat = sympy.Quaternion(cosT2,0,0,sinT2)*sympy.Quaternion(cosA2,sinA2,0,0)

class DenavitRow(Denavit):
    """
    This class defines a row in the Denavit Hartenberg Table
    """
    def __init__(self, dh_theta:float, dh_d:float, dh_a:float, dh_alfa:float, joint:Joint = None):
        super().__init__()
        self.dhParams = (dh_theta,dh_d,dh_a,dh_alfa)
        self.joint = joint

        if joint is not None:
            if joint.type is JointType.ROTATIONAL:
                T = self.denavit.subs(self.theta, self.theta+joint.symbol)
            elif joint.type is JointType.PRISMATIC:
                T = self.denavit.subs(self.d, self.d+joint.symbol)
            else:
                raise ValueError("Invalid joint type")
        else:
            T = self.denavit
        
        T = T.subs(self.theta,dh_theta)
        T = T.subs(self.d,dh_d)
        T = T.subs(self.a,dh_a)
        T = T.subs(self.alfa,dh_alfa)
        self.TransformSym = T
        if joint is None:
            self.TransformLambda = sympy.lambdify(sympy.Symbol('x'),T)
        else:
            self.TransformLambda = sympy.lambdify(joint.symbol,T) 

    def get(self):
        return (self.theta,self.d, self.a, self.alfa, self.joint.value)

    def eval(self, jointVal:float):
        self.joint.value = jointVal
        return self.TransformLambda(self.joint.value)
    
    def getRotationSym(self):
        return self.TransformSym[0:3,0:3]
    
    def getTranslationSym(self):
        return self.TransformSym[0:3,3]

class URDFMaterials():
    """
    This class includes the different materials to be used in URDF materials
    """
    RED     = 0
    GREEN   = 1
    BLUE    = 2
    CYAN    = 3
    MAGENTA = 4
    YELLOW  = 5
    BLACK   = 6
    COLORS_NUM = 7
    
    def __init__(self) -> None:
        self.materialStrings = {
            self.RED:\
                '\t\t\t<material name="red">\n'+\
                '\t\t\t\t<color rgba="1 0 0 1.0"/>\n'+\
                '\t\t\t</material>\n',
            self.GREEN:\
                '\t\t\t<material name="green">\n'+\
                '\t\t\t\t<color rgba="0 1 0 1.0"/>\n'+\
                '\t\t\t</material>\n',
            self.BLUE:\
                '\t\t\t<material name="blue">\n'+\
                '\t\t\t\t<color rgba="0 0 1 1.0"/>\n'+\
                '\t\t\t</material>\n',
            self.CYAN:\
                '\t\t\t<material name="cyan">\n'+\
                '\t\t\t\t<color rgba="0 1 1 1.0"/>\n'+\
                '\t\t\t</material>\n',
            self.MAGENTA:\
                '\t\t\t<material name="magenta">\n'+\
                '\t\t\t\t<color rgba="1 0 1 1.0"/>\n'+\
                '\t\t\t</material>\n',
            self.YELLOW:\
                '\t\t\t<material name="yellow">\n'+\
                '\t\t\t\t<color rgba="1 1 0 1.0"/>\n'+\
                '\t\t\t</material>\n',
            self.BLACK:\
                '\t\t\t<material name="black">\n'+\
                '\t\t\t\t<color rgba="0 0 0 1.0"/>\n'+\
                '\t\t\t</material>\n',
        }
        self.nextColor = self.RED

    def getMaterialStr(self, color):
        return self.materialStrings[color]

    def getCurrentMaterialStr(self):
        return self.getMaterialStr(self.nextColor)

    def getNextMaterialStr(self):
        color = self.nextColor
        self.nextColor = (1+color)%self.COLORS_NUM
        return self.getMaterialStr(color)

class DenavitDK:
    """
    Robot definiton from Denavit Hartenberg Table
    """
    def __init__(self,denavitRows, robotName:str = None, jacobianOrientation:OrientationType = OrientationType.MATRIX) -> None:
        self.denavitRows = denavitRows
        self.directTransformSym = sympy.Matrix(np.eye(4))
        self.jointsSym = []
        # Compute direct kinematis and record joint symbols
        for T in denavitRows:
            if T.joint is not None:
                self.jointsSym.append(T.joint.symbol)
            self.directTransformSym = self.directTransformSym*T.TransformSym
            # Clean almost zero values
            self.directTransformSym = sympy.nsimplify(self.directTransformSym,tolerance=1e-12,rational=True)
        # Operate fractions
        self.directTransformSym = self.directTransformSym.evalf()
        # self.directTransformSym = sympy.simplify(self.directTransformSym)
        # Set joints number
        self.jointsNum = len(self.jointsSym)
        # Set the robot name
        self.name = robotName if robotName is not None else f"{self.jointsNum}DOF_robot"
        # Set joints array
        self.jointsSym = sympy.Matrix([q for q in self.jointsSym])
        # Set lambda for direct transform
        self.directLambdaTransform = sympy.lambdify(self.jointsSym,self.directTransformSym)
        # Set the jacobian orientation
        self.jacobianOriType = jacobianOrientation
        # Calculate jacobian
        self.jacobian = self._jacobian(jacobianOrientation)
        if self.jacobian is None:
            self.jacobianLambda = None
            self.jacobianPos = None
            self.jacobianPosLambda = None
        else:
            self.jacobianLambda = sympy.lambdify(self.jointsSym,self.jacobian)
            # Calculate position only jacobian
            self.jacobianPos = self.jacobian[0:3,:]
            self.jacobianPosLambda = sympy.lambdify(self.jointsSym,self.jacobianPos)
        # Store zero position
        self.zeroPose = self.eval(np.zeros(self.jointsNum))

    def eval(self, jointVal:list):
        return self.directLambdaTransform(*jointVal)

    def getRotationSym(self):
        return self.directTransformSym[0:3,0:3]
    
    def getTranslationSym(self):
        return self.directTransformSym[0:3,3]
    
    def _jacobian(self, orientation:OrientationType = OrientationType.EULER):
        """
        Compute the Analitical Jacobian of the forward kinematic transformation

        #param orientation: modify with what type of orientation representation 
            the jacobian is built. Choices:
            - EULER: uses xyz and euler angles xyz
            - MATRIX: uses xyz and n and o vectors of the rotation matrix
            - QUATERNION: uses xyz and a quaternion
        """
        if 0 == self.jointsNum: return None
        rot = self.getRotationSym()
        pos = self.getTranslationSym()
        if   OrientationType.NONE == orientation:
            expression = pos
        elif OrientationType.EULER ==  orientation:
            expression = Rotations().matrixToEulerXYZSym(self.directTransformSym)
        elif OrientationType.MATRIX ==  orientation:
            expression = sympy.Matrix([pos, rot[0:3,0], rot[0:3,1] ])
        elif OrientationType.QUATERNION == orientation:
            quat = sympy.Quaternion.from_rotation_matrix(rot)
            expression = sympy.Matrix([pos, quat.a, quat.b, quat.c, quat.d])
        else:
            raise Exception("Unknown orientation type")
        return expression.jacobian(self.jointsSym)

    def inversePositionEval(self, init_joint_pose, end_pose, iterations=100, tolerance=1e-9):
        joints  = np.transpose(np.array([init_joint_pose]))
        endPose = np.transpose(np.array([end_pose]))
        
        for it in range(iterations):
            # Extract current pose as HTM
            currentPoseMatrix = self.eval(joints.ravel())
            # Retrive position
            subs = {sym:val for sym,val in zip(self.jointsSym,joints)}
            currentPose = sympy.Matrix([currentPoseMatrix[0:3,3]]).evalf(subs = subs)
            # Calculate difference and error
            delta = endPose - np.array(currentPose).astype(np.float64).transpose()
            error = np.linalg.norm(delta)**2
            print(f"[{it}] {error}")
            if error < tolerance:
                print(f"Solution with error {error} in {it} iterations")
                return joints.ravel()

            # Compute jacobian, invert it and compute the delta in joints
            currentJacobian = self.jacobianPosLambda(*joints.ravel())
            currentJacobianInv = np.linalg.pinv(currentJacobian)
            deltaJoints = np.matmul(currentJacobianInv,delta)

            joints = joints + deltaJoints
            
        print(f"Failed to compute positional inverse kinematics (Error: {error})")
        return init_joint_pose

    def inverseEval(self, init_joint_pose, end_pose, iterations=100, tolerance=1e-9):
        # Adjust position
        joints = self.inversePositionEval(init_joint_pose, end_pose[0:3,3])
        # Adjust orientation
        pos = sympy.Matrix(end_pose[0:3,3])
        rot = sympy.Matrix(end_pose[0:3,0:3])
        joints = np.transpose(np.array([joints]))
        if   OrientationType.NONE == self.jacobianOriType:
            return list(joints.ravel())
        elif OrientationType.EULER == self.jacobianOriType:
            endPose = Rotations().matrixToEulerXYZ(end_pose)
        elif OrientationType.MATRIX == self.jacobianOriType:
            endPose = sympy.Matrix([pos, rot[0:3,0], rot[0:3,1] ])
        elif OrientationType.QUATERNION == self.jacobianOriType:
            quat = sympy.Quaternion.from_rotation_matrix(rot)
            endPose = sympy.Matrix([pos, quat.a, quat.b, quat.c, quat.d ])
        endPose = np.array(endPose).astype(np.float64)

        for it in range(iterations):
            # Extract current pose as HTM
            currentPoseMatrix = self.eval(joints.ravel())
            # Adjust orientation type
            pos = sympy.Matrix(currentPoseMatrix[0:3,3])
            rot = sympy.Matrix(currentPoseMatrix[0:3,0:3])
            if   OrientationType.EULER == self.jacobianOriType:
                currentPose = Rotations().matrixToEulerXYZ(currentPoseMatrix)
            elif OrientationType.MATRIX == self.jacobianOriType:
                currentPose = sympy.Matrix([pos, rot[0:3,0], rot[0:3,1] ])
            elif OrientationType.QUATERNION == self.jacobianOriType:
                quat = sympy.Quaternion.from_rotation_matrix(rot)
                currentPose = sympy.Matrix([pos, quat.a, quat.b, quat.c, quat.d ])
            currentPose = np.array(currentPose).astype(np.float64)
            # Calculate difference and error
            delta = endPose - currentPose
            error = np.linalg.norm(delta)**2
            print(f"[{it}] {error}")
            if error < tolerance:
                print(f"Solution with error {error} in {it} iterations")
                return joints.ravel()

            # Compute jacobian, invert it and compute the delta in joints
            currentJacobian = self.jacobianLambda(*joints.ravel())
            currentJacobianInv = np.linalg.pinv(currentJacobian)
            deltaJoints = np.matmul(currentJacobianInv,delta)

            joints = joints + deltaJoints
            
        print(f"Failed to compute inverse kinematics (Error: {error})")
        return init_joint_pose

    def genCCode(self, filename:str=None, simplify:bool = False):
        from sympy.utilities.codegen import codegen
        if filename is None:
            filename = self.name

        fileExpressions = [
            ('directKin',  sympy.simplify(self.directTransformSym) if simplify else self.directTransformSym ),
            ('jacobianPos',sympy.simplify(self.jacobianPos)        if simplify else self.jacobianPos ),
            ('jacobian',   sympy.simplify(self.jacobian)           if simplify else self.jacobian ),
        ]

        [(c_name, c_code), _] = codegen(fileExpressions, "C99", filename, header=False, empty=False)
        with open(c_name,'w+') as c_file:
            c_file.write(c_code)
    
    def genURDF(self, filename:str=None):
        if filename is None:
            filename = self.name + ".urdf"
        elif not filename.endswith(".urdf"):
            filename += ".urdf"

        # Compute robot scale
        scale = 0
        for dr in self.denavitRows:
            _,d,a,_ = dr.dhParams
            scale += np.abs(d) + np.abs(a)
        scale *= 0.015

        with open(filename,'w+') as robfile:
            links = []
            # File heading
            robfile.write("<?xml version='1.0'?>\n\n")
            robfile.write("<!-- URDF file generated with KinematicsGenerator package -->\n")
            robfile.write("<!-- https://github.com/gmescudero/KinematicsGenerator -->\n")
            robfile.write(f"<robot name=\"{self.name}\">\n")
            # Add world link
            robfile.write(self._genURDFWorldLink(links,scale))
            # Add links
            robfile.write(self._genURDFLinks(links,scale))
            # Add joints
            robfile.write(self._genURDFJoints(links))
            # File ending
            robfile.write("</robot>")

    def _genURDFWorldLink(self,links:list,scale:float) -> str:
        string =  '\t<!-- ******************* WORLD LINK ******************* -->\n'
        # Add world and base links
        string +=  '\t<link name="world"/>\n'
        string +=  '\t<link name="base_joint">\n'
        string +=  '\t\t<origin xyz="0 0 0" rpy="0 0 0" />\n'
        string +=  '\t\t<visual>\n'
        string +=  '\t\t\t<geometry>\n'
        string += f'\t\t\t\t<cylinder radius=\"{scale*8}\" length=\"{scale}\" />\n'
        string +=  '\t\t\t</geometry>\n'
        string +=  '\t\t</visual>\n'
        string +=  '\t\t<inertial>\n'
        string += f'\t\t\t<mass value="0.0" />\n'
        string += f'\t\t\t<inertia ixx="0.0" ixy="0.0" ixz="0.0"  iyy="0.0" iyz="0.0" izz="0.0" />\n'
        string +=  '\t\t</inertial>\n'
        string += '\t</link>\n'
        links.append('base_joint')
        # Add base joint fixed to world
        string +=  '\t<joint name="joint_world" type="fixed">\n'
        string +=  '\t\t<parent link="world"/>\n'
        string += f'\t\t<child link="{links[0]}"/>\n'
        string +=  '\t\t<origin rpy="0 0 0" xyz="0 0 0"/>\n'
        string +=  '\t</joint>\n'
        return string

    def _genURDFLinks(self,links:list,scale:float) -> str:
        string = ''
        density = 1
        linksNum = 1
        material = URDFMaterials()
        material_str = material.getCurrentMaterialStr()
        radius = scale
        radius_delta = radius/(len(self.denavitRows)+1)
        for dr in self.denavitRows:
            _,d,a,_ = dr.dhParams
            link = np.sqrt(d**2 + a**2)
            if link > 1e-9 and dr.joint is not None: radius -= radius_delta
            mass = density*link*pi*radius**2
            name = f"L{linksNum}"
            links.append(name+"_joint")
            linksNum += 1
            if dr.joint is not None:
                material_str = material.getNextMaterialStr()

            string += f'\t<!-- ******************* {name} LINK ******************* -->\n'
            ## Joint visual
            string += f'\t<link name="{name}_joint">\n'
            if dr.joint is not None:
                string +=  '\t\t<visual>\n'
                string += f'\t\t\t<origin xyz="0 0 0" rpy="0 0 0" />\n'
                string +=  '\t\t\t<geometry>\n'
                if dr.joint.type is JointType.PRISMATIC:
                    string += f'\t\t\t\t<box size="{radius*3.0} {radius*3.0} {radius*8}"/>\n'
                else:
                    string += f'\t\t\t\t<cylinder radius=\"{radius*2}\" length=\"{radius*8}\" />\n'
                string +=  '\t\t\t</geometry>\n'
                string += material_str
                string +=  '\t\t</visual>\n'
                string +=  '\t\t<inertial>\n'
                string += f'\t\t\t<mass value="0.0" />\n'
                string += f'\t\t\t<inertia ixx="0.0" ixy="0.0" ixz="0.0"  iyy="0.0" iyz="0.0" izz="0.0" />\n'
                string +=  '\t\t</inertial>\n'
            string +=  '\t</link>\n'

            ## Link visual
            if link > 1e-9:
                string += f'\t<link name="{name}">\n'
                # Visual section
                string +=  '\t\t<visual>\n'
                string += f'\t\t\t<origin xyz="0 0 {link/2}" rpy="0 0 0" />\n'
                string +=  '\t\t\t<geometry>\n'
                string += f'\t\t\t\t<cylinder radius=\"{radius}\" length=\"{link}\" />\n'
                string +=  '\t\t\t</geometry>\n'
                string += material_str
                string +=  '\t\t</visual>\n'
                # Collision section
                string +=  '\t\t<collision>\n'
                string += f'\t\t\t<origin xyz="0 0 {link/2}" rpy="0 0 0" />\n'
                string +=  '\t\t\t<geometry>\n'
                # make collision length a little shorter to avoid self colliding
                string += f'\t\t\t\t<cylinder radius=\"{radius}\" length=\"{link-(radius*2.1)}\" />\n'
                string +=  '\t\t\t</geometry>\n'
                string +=  '\t\t</collision>\n'
                # Inertial section
                ixx = (1/12)*mass*(3*radius**2 + link**2)
                iyy = (1/12)*mass*(3*radius**2 + link**2)
                izz = (1/12)*mass*link**2
                string +=  '\t\t<inertial>\n'
                string += f'\t\t\t<mass value=\"{mass}\" />\n'
                string += f'\t\t\t<inertia ixx="{ixx}" ixy="0.0" ixz="0.0"  iyy="{iyy}" iyz="0.0" izz="{izz}" />\n'
                string +=  '\t\t</inertial>\n'

                string +=  '\t</link>\n'

                # Join the two visuals
                if   1e-6 > np.abs(link):
                    angle = 0
                elif 1e-6 > np.abs(d) and 0 < a:
                    angle = pi/2
                elif 1e-6 > np.abs(d) and 0 > a:
                    angle = -pi/2
                elif 1e-6 > a and 0 < d:
                    angle = 0
                elif 1e-6 > a and 0 > d:
                    angle = pi
                else:
                    angle = np.arctan2(a/link,d/link)
                string += f'\t<joint name="{name}_joint" type="fixed">\n'
                string += f'\t\t<origin xyz="0 0 0" rpy="0 {angle} 0"/>\n'
                string += f'\t\t<parent link="{name}_joint"/>\n'
                string += f'\t\t<child link="{name}"/>\n'
                string +=  '\t</joint>\n'

        string += '\n'
        return string

    def _genURDFJoints(self,links:list) -> str:
        string = '\t<!-- ********************* JOINTS ********************* -->\n'
        # Add joints from Denavit Hartenberg table
        prev_d = 0
        prev_a = 0
        prev_af = 0
        for i,dr in enumerate(self.denavitRows):
            th,d,a,af = dr.dhParams
            eulerXYZ = Rotations().matrixToEulerXYZSym(sympy.rot_axis1(prev_af)*sympy.rot_axis3(th))
            
            if dr.joint is None:
                string += f'\t<joint name="{prev_dr.joint.name}_" type="fixed">\n'
            elif dr.joint.type == JointType.ROTATIONAL:
                string += f'\t<joint name="{dr.joint.name}" type="revolute">\n'
                string += f'\t\t<limit lower="{dr.joint.lower_limit}" upper="{dr.joint.upper_limit}" effort="12" velocity="2.443"/>\n'
                string +=  '\t\t<axis xyz="0 0 1" />\n'
            else:
                string += f'\t<joint name="{dr.joint.name}" type="prismatic">\n'
                string += f'\t\t<limit lower="{dr.joint.lower_limit}" upper="{dr.joint.upper_limit}" effort="12" velocity="2.443"/>\n'
                string +=  '\t\t<axis xyz="0 0 1" />\n'
            string += f'\t\t<parent link="{links[i]}" />\n'
            string += f'\t\t<child link="{links[i+1]}" />\n'
            string += f'\t\t<origin xyz="{prev_a} 0 {prev_d}" rpy="{eulerXYZ[0].evalf()} {eulerXYZ[1].evalf()} {eulerXYZ[2].evalf()}" />\n'
            string += '\t</joint>\n'
            prev_dr = dr
            prev_d = d
            prev_a = a
            prev_af = af
        string += '\n'
        return string


class TwoLinkPlanar:

    def __init__(self, a1=None,a2=None,q1=None,q2=None,x=None,y=None) -> None:
        self.a1 = sympy.Symbol('a_1')
        self.a2 = sympy.Symbol('a_2')
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.x  = sympy.Symbol('x')
        self.y  = sympy.Symbol('y')
        self.theta = sympy.Symbol('theta')
        # Compute direct and inverse with default symbols
        self._directSymBaseEuler = self.directKinSymbols()
        self._directSymBase  = self.directKinHomoTSymbols()
        self._inverseSymBase = self.inverseKinSymbols()
        # Compute direct and inverse kinematics with custom symbols
        self.setSymbols(a1,a2,q1,q2,x,y)

    def directKinSymbols(self):
        x_dk     = self.a1*sympy.cos(self.q1) + self.a2*sympy.cos(self.q1+self.q2)
        y_dk     = self.a1*sympy.sin(self.q1) + self.a2*sympy.sin(self.q1+self.q2)
        theta_dk = self.q1 + self.q2
        T = sympy.Matrix((x_dk,y_dk,0,theta_dk,0,0))
        return T

    def directKinHomoTSymbols(self):
        x_dk,y_dk,z_dk,yaw_dk,pitch_dk,roll_dk = self.directKinSymbols()
        rotM = Rotations().zyxToMatrixSym(yaw_dk,pitch_dk,roll_dk)
        translation = sympy.Matrix((x_dk,y_dk,z_dk))
        T = sympy.Matrix((
            (rotM            , translation    ),
            (sympy.zeros(1,3), sympy.ones(1,1))
        ))
        return T

    def inverseKinSymbols(self):
        q2_ik = sympy.acos((self.x**2 + self.y**2 - self.a1**2 - self.a2**2)/(2*self.a1*self.a2))
        q1_ik = sympy.atan2(self.y,self.x) - sympy.atan2(self.a2*sympy.sin(q2_ik), self.a1 + self.a2*sympy.cos(q2_ik))
        return sympy.Matrix((q1_ik,q2_ik))

    def setSymbols(self,a1=None,a2=None,q1=None,q2=None,x=None,y=None):
        self._substitutions = {
            self.a1:a1 if a1 is not None else self.a1, 
            self.a2:a2 if a2 is not None else self.a2,
            self.q1:q1 if q1 is not None else self.q1, 
            self.q2:q2 if q2 is not None else self.q2,
            self.x:x if x is not None else self.x, 
            self.y:y if y is not None else self.y
        }
        self.directSymEuler = self._directSymBaseEuler.subs(self._substitutions,simultaneous=True)
        self.directSym  = self._directSymBase.subs(self._substitutions,simultaneous=True)
        self.inverseSym = self._inverseSymBase.subs(self._substitutions,simultaneous=True)

        # Clean almost zero values
        self.directSym = sympy.nsimplify(self.directSym,tolerance=1e-12,rational=True)
        self.directSym = self.directSym.evalf()
        self.inverseSym = sympy.nsimplify(self.inverseSym,tolerance=1e-12,rational=True)
        self.inverseSym = self.inverseSym.evalf()

        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify((self.q1,self.q2),self.directSym)
            self.inverseLambda = sympy.lambdify((self.x,self.y),self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2: None
            self.inverseLambda = lambda x,y: None

class TwoLinkAndBase():

    def __init__(self, a1=None,a2=None,q1=None,q2=None,q3=None,x=None,y=None,z=None) -> None:
        self.a1 = sympy.Symbol('a_1') 
        self.a2 = sympy.Symbol('a_2')
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.q3 = sympy.Symbol('q_3')
        self.x = sympy.Symbol('x')
        self.y = sympy.Symbol('y')
        self.z = sympy.Symbol('z')
        self._r = sympy.sqrt(self.x**2 + self.y**2)
        self.twoLink = TwoLinkPlanar(self.a1,self.a2,self.q2,self.q3,self._r,self.z)
        # Compute direct and inverse with default symbols
        self._directSymBase  = self.directKinHomoTSymbols()
        self._inverseSymBase = self.inverseKinSymbols()
        # Compute direct and inverse kinematics with custom symbols
        self.setSymbols(a1,a2,q1,q2,q3,x,y,z) 


    def setSymbols(self,a1=None,a2=None,q1=None,q2=None,q3=None,x=None,y=None,z=None):
        self._substitutions = {
            self.a1:a1 if a1 is not None else self.a1, 
            self.a2:a2 if a2 is not None else self.a2,
            self.q1:q1 if q1 is not None else self.q1, 
            self.q2:q2 if q2 is not None else self.q2, 
            self.q3:q3 if q3 is not None else self.q3,
            self.x:x if x is not None else self.x, 
            self.y:y if y is not None else self.y,
            self.z:z if z is not None else self.z
        }
        self._r = sympy.sqrt(self.x**2 + self.y**2).subs(self._substitutions)
        self.twoLink.setSymbols(self._substitutions[self.a1], self._substitutions[self.a2], 
                                self._substitutions[self.q2], self._substitutions[self.q3],
                                self._r, self._substitutions[self.z] )

        self.directSym  = self._directSymBase.subs(self._substitutions,simultaneous=True)
        self.inverseSym = self._inverseSymBase.subs(self._substitutions,simultaneous=True)
            
        # Clean almost zero values
        self.directSym = sympy.nsimplify(self.directSym,tolerance=1e-12,rational=True)
        self.directSym = self.directSym.evalf()
        self.inverseSym = sympy.nsimplify(self.inverseSym,tolerance=1e-12,rational=True)
        self.inverseSym = self.inverseSym.evalf()

        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify((self.q1,self.q2,self.q3),self.directSym)
            self.inverseLambda = sympy.lambdify((self.x,self.y,self.z),self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2,q3: None
            self.inverseLambda = lambda x,y,z: None
        
    def directKinSymbols(self):
        r_dk,z_dk,_,pitch_dk,_,_ = self.twoLink.directSymEuler
        x_dk = r_dk*sympy.cos(self.q1)
        y_dk = r_dk*sympy.sin(self.q1)
        yaw_dk = self.q1
        roll_dk = 0
        T = sympy.Matrix((x_dk,y_dk,z_dk,yaw_dk,pitch_dk,roll_dk))
        return T

    def directKinHomoTSymbols(self):
        x_dk,y_dk,z_dk,yaw_dk,pitch_dk,roll_dk = self.directKinSymbols()
        rotM = Rotations().zyxToMatrixSym(yaw_dk,pitch_dk,roll_dk)
        translation = sympy.Matrix((x_dk,y_dk,z_dk))
        T = sympy.Matrix((
            (rotM            , translation    ),
            (sympy.zeros(1,3), sympy.ones(1,1))
        ))
        return T

    def inverseKinSymbols(self):
        q1_ik = sympy.atan2(self.y,self.x)
        q2_ik,q3_ik = self.twoLink.inverseSym
        return sympy.Matrix((q1_ik,q2_ik,q3_ik))


class SphericalWrist:
    def __init__(self, sequence:str, q1=None,q2=None,q3=None,yaw=None,pitch=None,roll=None) -> None:
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.q3 = sympy.Symbol('q_3')
        self.yaw   = sympy.Symbol('gamma')
        self.pitch = sympy.Symbol('beta')
        self.roll  = sympy.Symbol('alpha')
        self.hmt   = sympy.MatrixSymbol('R',4,4)
        self.sequence = sequence
        self.singularities = {}
        # Compute direct and inverse with default symbols
        self._directSymBase  = self.directKinHomoTSymbols()
        self._inverseSymBase = self.inverseKinSymbols()
        # Compute direct and inverse kinematics with custom symbols
        self.setSymbols(q1,q2,q3,yaw,pitch,roll) 

    def setSymbols(self,q1=None,q2=None,q3=None,x=None,y=None,z=None):
        substitutions = {
            self.q1:q1 if q1 is not None else self.q1, 
            self.q2:q2 if q2 is not None else self.q2, 
            self.q3:q3 if q3 is not None else self.q3,
            self.yaw  :x if x is not None else self.yaw  , 
            self.pitch:y if y is not None else self.pitch,
            self.roll :z if z is not None else self.roll 
        }
        self.directSym  = self._directSymBase.subs(substitutions,simultaneous=True)
        self.inverseSym = self._inverseSymBase.subs(substitutions,simultaneous=True)

        if q1 is not None:
            self.inverseSym[0] = q1.subs({self.q1:self.inverseSym[0]})
        if q2 is not None:
            self.inverseSym[1] = q2.subs({self.q2:self.inverseSym[1]})
        if q3 is not None:
            self.inverseSym[2] = q3.subs({self.q3:self.inverseSym[2]})
            
        # Clean almost zero values
        self.directSym = sympy.nsimplify(self.directSym,tolerance=1e-12,rational=True)
        self.directSym = self.directSym.evalf() 
        self.inverseSym = sympy.nsimplify(self.inverseSym,tolerance=1e-12,rational=True)
        self.inverseSym = self.inverseSym.evalf()

        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify((self.q1,self.q2,self.q3),self.directSym)
            self.inverseLambda = sympy.lambdify((self.hmt),self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2,q3: None
            self.inverseLambda = lambda hmt: None


    def directKinHomoTSymbols(self):
        self.rotM = Rotations().eulerToMatrixSequenceSym(self.sequence,self.q1,self.q2,self.q3)
        return sympy.Matrix((
            (self.rotM       , sympy.zeros(3,1) ),
            (sympy.zeros(1,3), sympy.ones(1,1)  )
        ))
    
    def inverseKinSymbols(self):
        htm = sympy.MatrixSymbol('R',4,4)
        return Rotations().matrixToEulerSequenceSym(self.sequence,htm)

    def direct(self,q1,q2,q3):
        return self.directLambda(q1,q2,q3)
    
    def inverse(self,hmt):
        return self.inverseLambda(hmt)


class Decoupled6DOF:
    def __init__(self,a1=None,a2=None,q1=None,q2=None,q3=None,q4=None,q5=None,q6=None,x=None,y=None,z=None,yaw=None,pitch=None,roll=None) -> None:
        self.a1 = sympy.Symbol('a_1')
        self.a2 = sympy.Symbol('a_2')
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.q3 = sympy.Symbol('q_3')
        self.q4 = sympy.Symbol('q_4')
        self.q5 = sympy.Symbol('q_5')
        self.q6 = sympy.Symbol('q_6')
        self.x = sympy.Symbol('x')
        self.y = sympy.Symbol('y')
        self.z = sympy.Symbol('z')
        self.yaw   = sympy.Symbol('gamma')
        self.pitch = sympy.Symbol('beta')
        self.roll  = sympy.Symbol('alpha')
        self.positionSide    = TwoLinkAndBase(a1,a2,self.q1,self.q2,self.q3,self.x,self.y,self.z)
        self.orientationSide = SphericalWrist("xyx")
        # Initialize symbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()   
        # Initialize lambdas
        self.directLambda  = lambda q1,q2,q3: None
        self.inverseLambda = lambda x, y, z : None
        # Compute symbolic kinematics and lambda functions
        self._resetSymbolsAndLambdas() # TODO: Improve this

    def directKinHomoTSymbols(self):
        return self.orientationSide.directKinHomoTSymbols() * self.positionSide.directKinHomoTSymbols()

    def inverseKinSymbols(self):
        positionKin    = self.positionSide.inverseKinSymbols()
        orientationFromPositon = sympy.Matrix(self.positionSide.directKinSymbols()[3:])
        orientationKin = self.orientationSide.inverseKinSymbols()
        orientationFromPositon = orientationFromPositon.subs(self.q1,-positionKin[0],simultaneous=True)
        orientationFromPositon = orientationFromPositon.subs(self.q2,-positionKin[1],simultaneous=True)
        orientationFromPositon = orientationFromPositon.subs(self.q3,-positionKin[2],simultaneous=True)
        
        return sympy.Matrix((positionKin,orientationFromPositon+orientationKin))
        
    def direct(self,q1,q2,q3,q4,q5,q6):
        return self.directLambda(q1,q2,q3,q4,q5,q6)

    def inverse(self,x,y,z,yaw,pitch,roll):
        return self.inverseLambda(x,y,z,yaw,pitch,roll)

    def _resetSymbolsAndLambdas(self):
        # Compute simbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        # self.inverseSym = self.inverseKinSymbols()
        # Compute direct kinematic symbols
        dkSymbols = self._orderedSymbolsFromExpr(self.directSym)
        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify(dkSymbols,self.directSym)
            self.inverseLambda = sympy.lambdify((self.x, self.y, self.z, self.yaw, self.pitch, self.roll), self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2,q3,q4,q5,q6: None
            self.inverseLambda = lambda x, y, z, yaw,pch,rol: None

    def _orderedSymbolsFromExpr(self,expression):
        names = [sym.name for sym in expression.free_symbols]
        names.sort()
        return [sympy.Symbol(n) for n in names]

if __name__ == "__main__" :
    arm  = 10
    farm = 5
    palm = 1
    fing = 1.2

    T_shz = DenavitRow(0    , 0    , 0    , -pi/2, Joint(sympy.Symbol('sh_z'),JointType.ROTATIONAL,upper_limit=165*pi/180, lower_limit=-pi/2))
    T_shy = DenavitRow(pi/2 , 0    , 0    , pi/2,  Joint(sympy.Symbol('sh_y'),JointType.ROTATIONAL,upper_limit=pi/2,       lower_limit=-pi/2))
    T_shx = DenavitRow(-pi/2, -arm , 0    , pi/2,  Joint(sympy.Symbol('sh_x'),JointType.ROTATIONAL,upper_limit=pi/2,       lower_limit=-pi/2))
    T_elz = DenavitRow(0    , 0    , 0    , -pi/2, Joint(sympy.Symbol('el_z'),JointType.ROTATIONAL,upper_limit=165*pi/180, lower_limit=0))
    T_elx = DenavitRow(0    , -farm, 0    , pi/2,  Joint(sympy.Symbol('el_x'),JointType.ROTATIONAL,upper_limit=pi/6,       lower_limit=-110*pi/180))
    T_wrz = DenavitRow(pi/2 , 0    , 0    , -pi/2, Joint(sympy.Symbol('wr_z'),JointType.ROTATIONAL,upper_limit=10*pi/180,  lower_limit=-pi/6))
    T_wry = DenavitRow(0    , 0    , -palm, 0,     Joint(sympy.Symbol('wr_y'),JointType.ROTATIONAL,upper_limit=pi/3,       lower_limit=-pi/2))
    T_hdy = DenavitRow(0    , 0    , -fing, 0,     Joint(sympy.Symbol('hd_y'),JointType.ROTATIONAL,upper_limit=5*pi/180,   lower_limit=-pi/2))

    T_arm = DenavitDK((T_shz,T_shy,T_shx,T_elz,T_elx,T_wrz,T_wry,T_hdy),"humanArm8")
    T_arm.genURDF()

    thumb = 0.8
    T_thb = DenavitRow(0, 0, -thumb, 0)

    T_arm5 = DenavitDK((T_shz,T_shy,T_shx,T_elz,T_elx,T_thb),"humanArm5")
    T_arm5.genURDF()

    

    