from cmath import pi
import numpy as np
import sympy
from sympy.abc import x,y,z
from dataclasses import dataclass
from enum import Enum

class Rotations:
    def __init__(self):
        self.alpha  = sympy.Symbol('alpha')
        self.beta  = sympy.Symbol('beta')
        self.gamma = sympy.Symbol('gamma')
        self.theta = sympy.Symbol('theta')
        self.zRotM = sympy.Matrix([
            [sympy.cos(self.theta),-sympy.sin(self.theta), 0],
            [sympy.sin(self.theta), sympy.cos(self.theta), 0],
            [0                    , 0                    , 1]
        ])
        self.yRotM = sympy.Matrix([
            [sympy.cos(self.theta), 0,-sympy.sin(self.theta)],
            [0                    , 1, 0                    ],
            [sympy.sin(self.theta), 0, sympy.cos(self.theta)]
        ])
        self.xRotM = sympy.Matrix([
            [1, 0                    , 0                    ],
            [0, sympy.cos(self.theta),-sympy.sin(self.theta)],
            [0, sympy.sin(self.theta), sympy.cos(self.theta)]
        ])
    
    def eulerToMatrixSequenceSym(self,sequence:str, a=sympy.abc.a,b=sympy.abc.b,c=sympy.abc.c):
        if len(sequence) != 3: return None
        rotSymbols = [a,b,c]
        rotM = sympy.eye(3)
        for axis,symbol in zip(list(sequence),rotSymbols):
            if axis is "x":
                rotM = rotM*self.xRotM.subs(self.theta,symbol)
            elif axis is "y":
                rotM = rotM*self.yRotM.subs(self.theta,symbol)
            elif axis is "z":
                rotM = rotM*self.zRotM.subs(self.theta,symbol)
        return rotM
    
    def zyxToMatrixSym(self,z,y,x):
        return self.eulerToMatrixSequenceSym("zyx",z,y,x)
    
    def xyzToMatrixSym(self,x,y,z):
        return self.eulerToMatrixSequenceSym("xyz",x,y,z)

    def matrixToEulerXYZSym(self, matrix):
        pos = matrix[0:3,3]
        R = matrix[0:3,0:3]
        sy = sympy.sqrt(R[0,0]**2 +  R[1,0]**2)
        x = sympy.atan2(R[2,1] , R[2,2])
        y = sympy.atan2(-R[2,0], sy)
        z = sympy.atan2(R[1,0], R[0,0])
        return sympy.Matrix([[pos[0]],[pos[1]],[pos[2]],[x],[y],[z]])
        
    def matrixToEulerXYZ(self, matrix):
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
        return np.transpose(np.array([[pos[0], pos[1], pos[2], x, y, z]]))

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
            self.TransformLambda = sympy.lambdify(x,T)
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
    def __init__(self,denavitRows, robotName:str = None) -> None:
        self.denavitRows = denavitRows
        self.directTransformSym = sympy.Matrix(np.eye(4))
        self.jointsSym = []
        # Compute direct kinematis and record joint symbols
        for T in denavitRows:
            if T.joint is not None:
                self.jointsSym.append(T.joint.symbol)
            self.directTransformSym = self.directTransformSym*T.TransformSym
        # Clean almost zero values
        self.directTransformSym = sympy.nsimplify(self.directTransformSym,tolerance=1e-15,rational=True)
        # self.directTransformSym = sympy.simplify(self.directTransformSym)
        # Set joints number
        self.jointsNum = len(self.jointsSym)
        # Set the robot name
        self.name = robotName if robotName is not None else f"{self.jointsNum}DOF_robot"
        # Set joints array
        self.jointsSym = sympy.Matrix([q for q in self.jointsSym])
        # Set lambda for direct transform
        self.directLambdaTransform = sympy.lambdify(self.jointsSym,self.directTransformSym)
        # Calculate jacobian
        self.jacobian = self._jacobian()
        self.jacobianLambda = sympy.lambdify(self.jointsSym,self.jacobian)
        self.jacobianPosLambda = sympy.lambdify(self.jointsSym,self.jacobian[0:3,0:self.jointsNum])

    def eval(self, jointVal:list):
        return self.directLambdaTransform(*jointVal)

    def getRotationSym(self):
        return self.directTransformSym[0:3,0:3]
    
    def getTranslationSym(self):
        return self.directTransformSym[0:3,3]
    
    def _jacobian(self):
        eulerTransofrm = Rotations().matrixToEulerXYZSym(self.directTransformSym)
        jacobian = eulerTransofrm.jacobian(self.jointsSym)
        return jacobian

        Transform = sympy.eye(4)
        z_vector = sympy.Matrix([[0],[0],[1]])
        J = []
        for T in self.denavitRows:
            if T.joint is None:
                Transform = Transform*T.TransformSym
                continue
            elif T.joint.type is JointType.PRISMATIC:
                translation = Transform[0:3,0:3]*z_vector
                J_row = sympy.Matrix([translation,sympy.zeros(3,1)]).T
            elif T.joint.type is JointType.ROTATIONAL:
                rotation    = Transform[0:3,0:3]*z_vector
                translation = rotation.cross(self.getTranslationSym()-Transform[0:3,3])
                J_row = sympy.Matrix([translation,rotation]).T
            else:
                raise ValueError("Invalid joint type")
            Transform = Transform*T.TransformSym

            J.append(J_row)
        self.jacobian = sympy.Matrix(J).T
        self.jacobian = sympy.nsimplify(self.jacobian,tolerance=1e-10,rational=True)
        return self.jacobian
    
    # def _jacobianPinv(self):
    #     sympy.singular_value_decomposition()

    def inversePositionEval(self, init_joint_pose, end_pose):
        iterations = 1000
        joints = np.transpose(np.array([init_joint_pose]))
        endPoseEuler= np.transpose(np.array([end_pose]))
        currentPoseEuler = Rotations().matrixToEulerXYZ(self.eval(init_joint_pose))[0:3]
        deltaPoseEuler = endPoseEuler - currentPoseEuler
        error = np.linalg.norm(deltaPoseEuler)**2
        while error > 1e-4 and iterations > 0:
            iterations -= 1

            currentJacobian = self.jacobianPosLambda(*joints.ravel())
            currentJacobianInv = np.linalg.pinv(currentJacobian)
            deltaJoints = np.matmul(currentJacobianInv,deltaPoseEuler)

            joints = joints + deltaJoints

            currentPoseEuler = Rotations().matrixToEulerXYZ(self.eval(joints.ravel()))[0:3]
            deltaPoseEuler = endPoseEuler - currentPoseEuler
            error = np.linalg.norm(deltaPoseEuler)**2
            print(error)
            
        print(f"Solution with error {error} in {1000-iterations} iterations")
        return joints.ravel()

    def inverseEval(self, init_joint_pose, end_pose):
        # Adjust position
        joints = self.inversePositionEval(init_joint_pose, end_pose[0:3,3])
        # Adjust orientation
        iterations = 1000
        joints = np.transpose(np.array([joints]))
        endPoseEuler = Rotations().matrixToEulerXYZ(end_pose)
        currentPoseEuler = Rotations().matrixToEulerXYZ(self.eval(init_joint_pose))
        deltaPoseEuler = endPoseEuler - currentPoseEuler
        error = np.linalg.norm(deltaPoseEuler)**2
        while error > 1e-4 and iterations > 0:
            iterations -= 1
            
            currentJacobian = self.jacobianLambda(*joints.ravel())
            currentJacobianInv = np.linalg.pinv(currentJacobian)
            deltaJoints = np.matmul(currentJacobianInv,deltaPoseEuler)

            joints = joints + deltaJoints
            joints = joints % (2*pi)

            currentPoseEuler = Rotations().matrixToEulerXYZ(self.eval(joints.ravel()))
            deltaPoseEuler = endPoseEuler - currentPoseEuler
            error = np.linalg.norm(deltaPoseEuler)**2
            print(error)
            
        print(f"Solution with error {error} in {1000-iterations} iterations")
        return list(joints.ravel())

    def genCCode(self, filename:str=None, simplify:bool = False):
        from sympy.utilities.codegen import codegen
        if filename is None:
            filename = self.name

        if simplify:
            symDK = sympy.simplify(self.directTransformSym)
            symJ = sympy.simplify(self.jacobian)
        else:
            symDK = self.directTransformSym
            symJ = self.jacobian

        [(c_name, c_code_dk), _] = codegen(('directKin', symDK), "C99", filename, header=False, empty=False)
        [(c_name, c_code_j), _]  = codegen(('jacobian',   symJ), "C99", filename, header=False, empty=False)
        with open(c_name,'w+') as c_file:
            c_file.write(c_code_dk)
            c_file.write(c_code_j)
    
    def genURDF(self, filename:str=None):
        if filename is None:
            filename = self.name + ".urdf"
        elif not filename.endswith(".urdf"):
            filename += ".urdf"

        with open(filename,'w+') as robfile:
            links = []
            # File heading
            robfile.write("<?xml version='1.0'?>\n\n")
            robfile.write("<!-- URDF file generated with KinematicsGenerator package -->\n")
            robfile.write("<!-- https://github.com/gmescudero/KinematicsGenerator -->\n")
            robfile.write(f"<robot name=\"{self.name}\">\n")
            # Add world link
            robfile.write(self._genURDFWorldLink(links))
            # Add links
            robfile.write(self._genURDFLinks(links))
            # Add joints
            robfile.write(self._genURDFJoints(links))
            # File ending
            robfile.write("</robot>")

    def _genURDFWorldLink(self,links:list) -> str:
        string =  '\t<!-- ******************* WORLD LINK ******************* -->\n'
        string += '\t<link name="world_joint">\n'
        string += '\t\t<origin xyz="0 0 0" rpy="0 0 0" />\n'
        string += '\t</link>\n'
        links.append('world')
        return string

    def _genURDFLinks(self,links:list) -> str:
        string = ''
        density = 1
        transM = np.eye(4)
        linksNum = 1
        material = URDFMaterials()
        material_str = material.getCurrentMaterialStr()
        zeroPose = self.eval(np.zeros(self.jointsNum))
        radius = np.linalg.norm(zeroPose[0:3])/70
        radius_delta = radius/(len(self.denavitRows)+1)
        for dr in self.denavitRows:
            th,d,a,af = dr.dhParams
            link = np.sqrt(dr.dhParams[1]**2 + dr.dhParams[2]**2)
            if link > 1e-9 and dr.joint is not None: radius -= radius_delta
            mass = density*link*pi*radius**2
            name = f"L{linksNum}"
            links.append(name)
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
                    string += f'\t\t\t\t<box size="{radius*1.5} {radius*1.5} {radius*8}"/>\n'
                else:
                    string += f'\t\t\t\t<cylinder radius=\"{radius*2}\" length=\"{radius*8}\" />\n'
                string +=  '\t\t\t</geometry>\n'
                string += material_str
                string +=  '\t\t</visual>\n'
            string +=  '\t</link>\n'

            ## Link visual
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
            string += f'\t\t\t<origin xyz="{link/2} 0 {link}" rpy="0 0 0" />\n'
            string +=  '\t\t\t<geometry>\n'
            string += f'\t\t\t\t<cylinder radius=\"{radius}\" length=\"{link}\" />\n'
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
            if   1e-6 > link:
                angle = 0
            elif 1e-6 > d:
                angle = pi/2
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
        prev_d = 0
        prev_a = 0
        prev_af = 0
        for i,dr in enumerate(self.denavitRows):
            th,d,a,af = dr.dhParams
            if dr.joint is None:
                string += f'\t<joint name="{prev_dr.joint.name}_" type="fixed">\n'
                string += f'\t\t<parent link="{links[i]}_joint" />\n'
                string += f'\t\t<child link="{links[i+1]}_joint" />\n'
                string += f'\t\t<origin xyz="{prev_a} 0 {prev_d}" rpy="{prev_af} 0 {th}" />\n'
            elif dr.joint.type == JointType.ROTATIONAL:
                string += f'\t<joint name="{dr.joint.name}" type="revolute">\n'
                string += f'\t\t<limit lower="{dr.joint.lower_limit}" upper="{dr.joint.upper_limit}" effort="12" velocity="2.443"/>\n'
                string += f'\t\t<parent link="{links[i]}_joint" />\n'
                string += f'\t\t<child link="{links[i+1]}_joint" />\n'
                string += f'\t\t<origin xyz="{prev_a} 0 {prev_d}" rpy="{prev_af} 0 {th}" />\n'
                string +=  '\t\t<axis xyz="0 0 1" />\n'
            else:
                string += f'\t<joint name="{dr.joint.name}" type="prismatic">\n'
                string += f'\t\t<limit lower="{dr.joint.lower_limit}" upper="{dr.joint.upper_limit}" effort="12" velocity="2.443"/>\n'
                string += f'\t\t<parent link="{links[i]}_joint" />\n'
                string += f'\t\t<child link="{links[i+1]}_joint" />\n'
                string += f'\t\t<origin xyz="{prev_a} 0 {prev_d}" rpy="{prev_af} 0 {th}" />\n'
                string +=  '\t\t<axis xyz="0 0 1" />\n'                

            string +=  '\t</joint>\n'
            prev_dr = dr
            prev_d = d
            prev_a = a
            prev_af = af
        string += '\n'
        return string


class TwoLinkPlanar:

    def __init__(self, a1=None, a2=None) -> None:
        self.a1 = sympy.Symbol('a_1') if a1 is None else a1
        self.a2 = sympy.Symbol('a_2') if a2 is None else a2
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.x  = x
        self.y  = y
        self.theta = sympy.Symbol('theta')
        self.directLambda  = lambda q1,q2: None
        self.inverseLambda = lambda x,y: None
        self._resetSymbolsAndLambdas()

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

    def setLengths(self,a1,a2):
        self.a1 = a1
        self.a2 = a2
        self._resetSymbolsAndLambdas()
    
    def setJointCoordinates(self,joint1,joint2):
        self.q1 = joint1
        self.q2 = joint2
        self._resetSymbolsAndLambdas()

    def setCartesianCoordinates(self,coord1,coord2):
        self.x = coord1
        self.y = coord2
        self._resetSymbolsAndLambdas()
    
    def _resetSymbolsAndLambdas(self):
        # Compute simbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()
        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify((self.q1,self.q2),self.directSym)
            self.inverseLambda = sympy.lambdify((self.x,self.y),self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2: None
            self.inverseLambda = lambda x,y: None

class TwoLinkAndBase():

    def __init__(self, a1=None, a2=None) -> None:
        self.a1 = sympy.Symbol('a_1') if a1 is None else a1
        self.a2 = sympy.Symbol('a_2') if a2 is None else a2
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.q3 = sympy.Symbol('q_3')
        self.x = x
        self.y = y
        self.z = z
        self._r = sympy.sqrt(self.x**2 + self.y**2)
        self.twoLink = TwoLinkPlanar(a1,a2)
        self.twoLink.setCartesianCoordinates(self._r, self.z)
        self.twoLink.setJointCoordinates(self.q2,self.q3)
        # Initialize symbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()   
        # Initialize lambdas
        self.directLambda  = lambda q1,q2,q3: None
        self.inverseLambda = lambda x, y, z : None
        # Compute symbolic kinematics and lambda functions
        self._resetSymbolsAndLambdas()

    def setLengths(self,a1,a2):
        self.a1 = a1
        self.a2 = a2
        self._resetSymbolsAndLambdas()
    
    def setJointCoordinates(self,joint1,joint2,joint3):
        self.q1 = joint1
        self.q2 = joint2
        self.q3 = joint3
        self._resetSymbolsAndLambdas()

    def setCartesianCoordinates(self,coord1,coord2,coord3):
        self.x = coord1
        self.y = coord2
        self.z = coord3
        self._resetSymbolsAndLambdas()
        
    def directKinSymbols(self):
        r_dk,z_dk,_,pitch_dk,_,_ = self.twoLink.directKinSymbols()
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
        q2_ik,q3_ik = self.twoLink.inverseKinSymbols()
        return sympy.Matrix((q1_ik,q2_ik,q3_ik))
        
    def _resetSymbolsAndLambdas(self):
        # Compute simbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()
        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify((self.q1,self.q2,self.q3),self.directSym)
            self.inverseLambda = sympy.lambdify((self.x, self.y, self.z), self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2,q3: None
            self.inverseLambda = lambda x, y, z : None


class SphericalWrist:
    def __init__(self, sequence:str) -> None:
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.q3 = sympy.Symbol('q_3')
        self.yaw   = sympy.Symbol('gamma')
        self.pitch = sympy.Symbol('beta')
        self.roll  = sympy.Symbol('alpha')
        self.sequence = sequence
        self.singularities = {}
        # Initialize symbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()   
        # Initialize lambdas
        self.directLambda  = lambda q1,q2,q3: None
        self.inverseLambda = lambda x, y, z : None
        # Cmpute symbolic kinematics and lambda functions
        self._resetSymbolsAndLambdas()

    def setJointCoordinates(self,joint1,joint2,joint3):
        self.q1 = joint1
        self.q2 = joint2
        self.q3 = joint3
        self._resetSymbolsAndLambdas()

    def setCartesianCoordinates(self,coord1,coord2,coord3):
        self.yaw   = coord1
        self.pitch = coord2
        self.roll  = coord3
        self._resetSymbolsAndLambdas()

    def directKinHomoTSymbols(self):
        self.rotM = Rotations().eulerToMatrixSequenceSym(self.sequence,self.q1,self.q2,self.q3)
        return sympy.Matrix((
            (self.rotM       , sympy.zeros(3,1) ),
            (sympy.zeros(1,3), sympy.ones(1,1)  )
        ))

    def inverseKinSymbols(self):
        rotM = Rotations().eulerToMatrixSequenceSym(self.sequence,self.yaw,self.pitch,self.roll)
        if self.sequence is "xyx":
            q1_dk = sympy.atan2(-rotM[1,0], rotM[2,0])
            q2_dk = sympy.acos(  rotM[0,0])
            q3_dk = sympy.atan2( rotM[0,1], rotM[0,2])
            self.singularities.update({self.pitch:0})
        else:
            return None

        return sympy.Matrix((q1_dk,q2_dk,q3_dk))

    def direct(self,q1,q2,q3):
        return self.directLambda(q1,q2,q3)
    
    def inverse(self,y,p,r):
        return self.inverseLambda(y,p,r)

    def _resetSymbolsAndLambdas(self):
        # Compute simbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()
        print(f"Singularities in {self.singularities}")
        # Compute direct kinematic symbols
        dkSymbols = self._orderedSymbolsFromExpr(self.directSym)
        # Compute lambda kinematic functions 
        try:
            self.directLambda  = sympy.lambdify(dkSymbols,self.directSym)
            self.inverseLambda = sympy.lambdify((self.yaw, self.pitch, self.roll), self.inverseSym)
        except SyntaxError:
            self.directLambda  = lambda q1,q2,q3: None
            self.inverseLambda = lambda x, y, z : None
    
    def _orderedSymbolsFromExpr(self,expression):
        names = [sym.name for sym in expression.free_symbols]
        names.sort()
        return [sympy.Symbol(n) for n in names]


class Decoupled6DOF:
    def __init__(self,a1=None,a2=None) -> None:
        self.a1 = sympy.Symbol('a_1') if a1 is None else a1
        self.a2 = sympy.Symbol('a_2') if a2 is None else a2
        self.q1 = sympy.Symbol('q_1')
        self.q2 = sympy.Symbol('q_2')
        self.q3 = sympy.Symbol('q_3')
        self.q4 = sympy.Symbol('q_4')
        self.q5 = sympy.Symbol('q_5')
        self.q6 = sympy.Symbol('q_6')
        self.x = x
        self.y = y
        self.z = z
        self.yaw   = sympy.Symbol('gamma')
        self.pitch = sympy.Symbol('beta')
        self.roll  = sympy.Symbol('alpha')
        self.positionSide    = TwoLinkAndBase(a1,a2)
        self.positionSide.setCartesianCoordinates(self.x,self.y,self.z)
        self.positionSide.setJointCoordinates(self.q1,self.q2,self.q3)
        self.orientationSide = SphericalWrist("xyx")
        self.orientationSide.setCartesianCoordinates(self.yaw,self.pitch,self.roll)
        self.orientationSide.setJointCoordinates(self.q4,self.q5,self.q6)
        # Initialize symbolic kinematics
        self.directSym  = self.directKinHomoTSymbols()
        self.inverseSym = self.inverseKinSymbols()   
        # Initialize lambdas
        self.directLambda  = lambda q1,q2,q3: None
        self.inverseLambda = lambda x, y, z : None
        # Compute symbolic kinematics and lambda functions
        self._resetSymbolsAndLambdas()

    def directKinHomoTSymbols(self):
        return self.orientationSide.directKinHomoTSymbols() * self.positionSide.directKinHomoTSymbols()

    def inverseKinSymbols(self):
        positionKin    = self.positionSide.inverseKinSymbols()
        orientationFromPositon = sympy.Matrix(self.positionSide.directKinSymbols()[3:])
        orientationKin = self.orientationSide.inverseKinSymbols()
        orientationFromPositon = orientationFromPositon.subs(self.q1,-positionKin[0])
        orientationFromPositon = orientationFromPositon.subs(self.q2,-positionKin[1])
        orientationFromPositon = orientationFromPositon.subs(self.q3,-positionKin[2])
        
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
    T1 = DenavitRow(pi/2,   0, 0, pi/2,  Joint(sympy.Symbol('q_1'),JointType.PRISMATIC))
    T2 = DenavitRow(pi/2,   0, 0, -pi/2, Joint(sympy.Symbol('q_2'),JointType.PRISMATIC))
    T3 = DenavitRow(0,      0, 0, 0,     Joint(sympy.Symbol('q_3'),JointType.PRISMATIC))

    T_cartesian = DenavitDK((T1,T2,T3),"Cartesian")
    print(T_cartesian.directTransformSym)
