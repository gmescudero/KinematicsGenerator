import os
import cloudpickle
from numpy import pi
import numpy as np
import sympy
from dataclasses import dataclass
from enum import IntEnum
import copy as cp
import csv
import argparse
from datetime import datetime


def printM(name:str, M):
    """
    Print a given matrix of 4 by 4
    """
    try: 
        M = sympy.nsimplify(M,tolerance=1e-12,rational=True).evalf()
    except Exception:
        pass

    print(f"{name}\t---------------------------------------\n"+
          f"> pos:\t{float(M[0,3]):.3f}\t{float(M[1,3]):.3f}\t{float(M[2,3]):.3f} \n"+
          f"> ori:\t{float(M[0,0]):.3f}\t{float(M[0,1]):.3f}\t{float(M[0,2]):.3f} \n"+
          f"      \t{float(M[1,0]):.3f}\t{float(M[1,1]):.3f}\t{float(M[1,2]):.3f} \n"+
          f"      \t{float(M[2,0]):.3f}\t{float(M[2,1]):.3f}\t{float(M[2,2]):.3f}")
    

class OrientationType(IntEnum):
    NONE = -1
    MATRIX = 0
    EULER = 1
    QUATERNION = 2


class Rotations:
    """
    A class for grouping common rotation tools
    """
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
        skewK = self.skewSymetric(self.k)
        self.rodriges = sympy.eye(3)*sympy.cos(self.alpha) + (1-sympy.cos(self.alpha))*(self.k*self.k.T) + sympy.sin(self.alpha)*skewK
        self.screwMatrix = sympy.Matrix([
            [self.rodriges,     (sympy.eye(3)-self.rodriges)*self.p],
            [sympy.zeros(1,3),  sympy.eye(1)                       ]
        ])
    
    def skewSymetric(self,vect):
        return sympy.Matrix([
            [ 0      ,-vect[2], vect[1]],
            [ vect[2], 0      ,-vect[0]],
            [-vect[1], vect[0], 0      ]
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
        if   sequence == "xyz":
            euler = self._m2eXYZ(rotM)
        elif sequence == "zyx":
            euler = self._m2eZYX(rotM)
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
        return sympy.Matrix(euler).evalf()[:,0]
    
    def _m2eXYZ(self,rotM):
        euler = [0,0,0]

        euler[0] = sympy.atan2(-rotM[1,2], rotM[2,2])
        euler[1] = sympy.asin(  rotM[0,2])
        euler[2] = sympy.atan2(-rotM[0,1], rotM[0,0])
        try:
            if np.abs(rotM[0,2]) >= 1.0 - 1e-12:
                sign = 1 if rotM[0,2] > 0 else -pi/2
                euler[0] = sympy.atan2(sign*rotM[1,0], sign*rotM[2,0])
                euler[1] = sign*pi/2 
                euler[2] = 0
        except TypeError:
            pass

        return euler

    def _m2eZYX(self,rotM,solution=1):
        euler = [0,0,0]

        euler[0] = sympy.atan2( rotM[1,0], rotM[0,0])                     # y
        euler[1] = (0 if solution == 1 else pi/2) - sympy.asin(rotM[2,0]) # p
        euler[2] = sympy.atan2( rotM[2,1], rotM[2,2])                     # r
        try:
            if np.abs(rotM[2,0]) >= 1.0 - 1e-12:
                euler[0] = 0
                sign = 1.0 if rotM[2,0] < 0 else -1.0
                euler[1] = sign*pi*0.5
                euler[2] = sympy.atan2( sign*rotM[0,1], sign*rotM[0,2])
        except TypeError:
            pass

        return euler

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

    def matrixToEulerRPYSym(self, matrix):
        if matrix.shape[0] == 4:
            pos = matrix[0:3,3]
        eulerZYX = self.matrixToEulerSequenceSym("zyx", matrix)
        if matrix.shape[0] == 4:
            return sympy.Matrix([pos[0],pos[1],pos[2], eulerZYX[2], eulerZYX[1], eulerZYX[0]]).evalf()
        return sympy.Matrix([eulerZYX[2], eulerZYX[1], eulerZYX[0]]).evalf()
    
    def matrixToEulerRPY(self, matrix):
        sym = self.matrixToEulerRPYSym(matrix)
        return np.array(sym)[:,0]
    
    def rotateVector(self, rotM, v):
        return [
            v[0]*rotM[0,0] + v[1]*rotM[0,1] + v[2]*rotM[0,2],
            v[0]*rotM[1,0] + v[1]*rotM[1,1] + v[2]*rotM[1,2],
            v[0]*rotM[2,0] + v[1]*rotM[2,1] + v[2]*rotM[2,2]
        ]

    def rotMatrixX(self, angle):
        return self.xRotM.subs(self.theta,angle).evalf()

    def rotMatrixY(self, angle):
        return self.yRotM.subs(self.theta,angle).evalf()

    def rotMatrixZ(self, angle):
        return self.zRotM.subs(self.theta,angle).evalf()
    
    def adjointT(self, T):
        R = sympy.Matrix(T[0:3,0:3])
        p = sympy.Matrix(T[0:3,3])
        skewP = self.skewSymetric(p)
        return sympy.Matrix([
            [R               , skewP*R],
            [sympy.zeros(3,3), R]
        ])
        # return sympy.Matrix([
        #     [R      , sympy.zeros(3,3)],
        #     [skewP*R, R]
        # ])


class JointType(IntEnum):
    """
    This Enums the joint types supported 
    """
    ROTATIONAL = 0
    PRISMATIC = 1

    def __repr__(self):
        strings = ["ROTATIONAL","PRISMATIC"]
        return strings[self.value]
    
    def __str__(self):
        return self.__repr__()

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
    
    def __eq__(self, value):
        if self.symbol != value.symbol: return False
        if self.type   != value.type:   return False
        if self.name   != value.name:   return False
        if not np.isclose(self.upper_limit, value.upper_limit): return False
        if not np.isclose(self.lower_limit, value.lower_limit): return False
        return True

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

    def __eq__(self, value):
        try:
            for i,p in enumerate(self.dhParams):
                if p != value.dhParams[i]:
                    return False
            return self.joint == value.joint
        except Exception as e:
            print(e)
            return False
    
    def __repr__(self):
        return f"DH: {self.dhParams} Joint: {self.joint}"

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
    def __init__(self,
                 denavitRows, 
                 robotName:str = None,
                 worldToBase:sympy.MutableDenseMatrix = None, 
                 tcpOffset:sympy.MutableDenseMatrix = None,
                 jacobianOrientation:OrientationType = OrientationType.MATRIX,
                 saveModelToFile:bool = False
                 ) -> None:
        
        self.worldToBase     = worldToBase
        self.tcpOffset       = tcpOffset
        self.denavitRows     = denavitRows
        self.jacobianOriType = jacobianOrientation
        self.name            = robotName if robotName is not None else f"{self.jointsNum}DOF_robot"
        self.jointsNum = 0
        for row in denavitRows:
            if row.joint is not None: self.jointsNum += 1

        # Load a model from file if possible
        model = self._loadModel()
        if model is not None:
            print(f"{datetime.now()}: Retrieved model from file {robotName}.pkl")
            self.__dict__ = model.__dict__
            return

        self.homogenousTransfromSym = sympy.eye(4) if worldToBase is None else worldToBase
        self.jointsSym = []
        # Compute direct kinematis and record joint symbols
        for T in denavitRows:
            if (DenavitRow is type(T)):
                if T.joint is not None:
                    self.jointsSym.append(T.joint.symbol)
                self.homogenousTransfromSym = self.homogenousTransfromSym*T.TransformSym
                # Clean almost zero values
                self.homogenousTransfromSym = sympy.nsimplify(self.homogenousTransfromSym,tolerance=1e-12,rational=True)
            else:
                self.homogenousTransfromSym = self.homogenousTransfromSym*T
        if tcpOffset is not None:
            self.homogenousTransfromSym = self.homogenousTransfromSym*tcpOffset
        # Operate fractions
        # from sympy.simplify.fu import TR0
        # self.homogenousTransfromSym = TR0(self.homogenousTransfromSym)
        self.homogenousTransfromSym = self.homogenousTransfromSym.evalf()
        # self.homogenousTransfromSym = sympy.simplify(self.homogenousTransfromSym)
        self.directTransformSym = sympy.Matrix([
            self.homogenousTransfromSym[3], self.homogenousTransfromSym[7], self.homogenousTransfromSym[11], 
            self.homogenousTransfromSym[0], self.homogenousTransfromSym[4], self.homogenousTransfromSym[8], 
            self.homogenousTransfromSym[1], self.homogenousTransfromSym[5], self.homogenousTransfromSym[9]
        ])
        # Set joints number
        self.jointsNum = len(self.jointsSym)
        # Set joints array
        self.jointsSym = sympy.Matrix([q for q in self.jointsSym])
        # Set lambda for direct transform
        try:
            self.directLambdaTransform = sympy.lambdify(self.jointsSym,self.directTransformSym)
        except NameError:
            self.directLambdaTransform = None
        # Calcualte geometrical jacobian
        self.jacobianGeom = self._jacobianGeometric()
        # Calculate analitical jacobian
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
            # Calculate orientation only jacobian
            self.jacobianOri = self.jacobian[3:,:]
            self.jacobianOriLambda = sympy.lambdify(self.jointsSym,self.jacobianOri)
        # Store zero position
        # if self.directLambdaTransform is not None:
        #     self.zeroPose = self.eval(np.zeros(self.jointsNum))
        if saveModelToFile:
            self._saveModel()

    def __eq__(self, other):
        if self.jointsNum != other.jointsNum: return False
        for i,row in enumerate(self.denavitRows):
            if row != other.denavitRows[i]:
                return False
        if self.worldToBase     != other.worldToBase: return False
        if self.tcpOffset       != other.tcpOffset: return False
        if self.jacobianOriType != other.jacobianOriType: return False
        return True

    def eval(self, jointVal:list):
        return np.array(self.directLambdaTransform(*jointVal)).astype(np.float64).flatten()

    def getRotationSym(self):
        return self.homogenousTransfromSym[0:3,0:3]
    
    def getTranslationSym(self):
        return self.homogenousTransfromSym[0:3,3]
    
    def _jacobianGeometric(self) -> sympy.MutableDenseMatrix:
        """
        Compute the Geometric jacobian from the Denavit Hartenberg table
        """
        jacobianGeom:sympy.MutableDenseMatrix = None
        zVector = sympy.Matrix([0,0,1])
        partialT = sympy.eye(4) if self.worldToBase is None else self.worldToBase
        translationTotal = self.getTranslationSym()
        for T in self.denavitRows:
            if (DenavitRow is type(T)):
                if T.joint is not None:
                    if T.joint.type == JointType.PRISMATIC:
                        jacobianColum = sympy.Matrix(sympy.BlockMatrix([
                            [partialT[0:3,0:3]*zVector],
                            [sympy.zeros(3,1)]
                        ]))
                    elif T.joint.type == JointType.ROTATIONAL:
                        rotatedZVector = partialT[0:3,0:3]*zVector
                        jacobianColum = sympy.Matrix(sympy.BlockMatrix([
                            [rotatedZVector.cross(translationTotal-partialT[0:3,3])],
                            [rotatedZVector]
                        ]))
                    jacobianColum = sympy.nsimplify(jacobianColum,tolerance=1e-12,rational=True).evalf()
                    if jacobianGeom is None:
                        jacobianGeom = jacobianColum
                    else:
                        jacobianGeom = jacobianGeom.col_insert(999,jacobianColum)
                # Add the transformation
                partialT = partialT*T.TransformSym
            else:
                partialT = partialT*T
        return jacobianGeom


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
            expression = sympy.Matrix([pos,Rotations().matrixToEulerSequenceSym("xyz",rot)])
        elif OrientationType.MATRIX ==  orientation:
            expression = sympy.Matrix([pos, rot[0:3,0], rot[0:3,1] ])
        elif OrientationType.QUATERNION == orientation:
            quat = sympy.Quaternion.from_rotation_matrix(rot)
            expression = sympy.Matrix([pos, quat.a, quat.b, quat.c, quat.d])
        else:
            raise Exception("Unknown orientation type")
        return expression.jacobian(self.jointsSym)
    

    def _adjustTaskSpaceInput(self, end_pose:np.ndarray, pos_only:bool = False):
        """
        Adjust the input pose to the standard format
        """
        if (4,4) == end_pose.shape:
            pos = np.array(end_pose[0:3,3]).astype(np.float64)
            rot = np.array(end_pose[0:3,0:3]).astype(np.float64)
            homoT = end_pose
            vectPose = np.array([
                pos[0]  , pos[1]  , pos[2],
                rot[0,0], rot[1,0], rot[2,0],
                rot[0,1], rot[1,1], rot[2,1]
            ]).astype(np.float64).transpose()
        elif 9 == len(end_pose):
            pos = np.array(end_pose[0:3]).astype(np.float64)
            v1 = np.array(end_pose[3:6]).astype(np.float64)
            v2 = np.array(end_pose[6:9]).astype(np.float64)
            v3 = np.cross(v1,v2)
            rot = np.array([v1,v2,v3]).astype(np.float64).transpose()
            homoT = np.array([
                [rot[0,0], rot[0,1], rot[0,2], pos[0]],
                [rot[1,0], rot[1,1], rot[1,2], pos[1]],
                [rot[2,0], rot[2,1], rot[2,2], pos[2]],
                [0, 0, 0, 1]
            ]).astype(np.float64)
            vectPose = end_pose
        elif 3 == len(end_pose):
            return end_pose
        else:
            raise Exception(f"Invalid pose: {end_pose.shape}")

        if pos_only:
            return pos
        elif OrientationType.NONE == self.jacobianOriType:
            return None
        elif OrientationType.EULER == self.jacobianOriType:
            return Rotations().matrixToEulerSequence("zyx",homoT)
        elif OrientationType.MATRIX == self.jacobianOriType:
            return vectPose
        elif OrientationType.QUATERNION == self.jacobianOriType:
            quat = sympy.Quaternion.from_rotation_matrix(rot)
            return sympy.Matrix([pos, quat.a, quat.b, quat.c, quat.d ])


    def inversePositionEval(self, init_joint_pose, end_pose:np.ndarray, iterations=100, tolerance=1e-9):
        """
        Compute the inverse kinematics of the robot just for the position
        """
        endPose = self._adjustTaskSpaceInput(end_pose, pos_only=True)
        joints  = np.transpose(np.array([init_joint_pose]).flatten())
        
        for it in range(iterations):
            # Extract current pose as HTM
            currentPoseMatrix = self.eval(joints.ravel())
            # Calculate difference and error
            delta = endPose - currentPoseMatrix[0:3]
            error = np.linalg.norm(delta)**2
            print(f"\t [{it}] {error}")
            if error < tolerance:
                print(f"{datetime.now()}: Solution with error {error} in {it} iterations")
                return joints.ravel()

            # Compute jacobian, invert it and compute the delta in joints
            currentJacobian = self.jacobianPosLambda(*joints.ravel())
            currentJacobianInv = np.linalg.pinv(currentJacobian)
            deltaJoints = np.matmul(currentJacobianInv,delta)

            joints = joints + deltaJoints
            
        print(f"{datetime.now()}: Failed to compute positional inverse kinematics (Error: {error})")
        return init_joint_pose


    def inverseEval(self, init_joint_pose, end_pose:np.ndarray, iterations=100, tolerance=1e-9):
        """
        Compute the inverse kinematics of the robot
        """
        # Adjust input to standard method
        endPose = self._adjustTaskSpaceInput(end_pose)

        # Adjust position
        joints = self.inversePositionEval(init_joint_pose, endPose)

        for it in range(iterations):
            # Extract current pose as HTM
            currentPoseMatrix = self.eval(joints.ravel())
            # Adjust orientation type
            currentPose = self._adjustTaskSpaceInput(currentPoseMatrix)
            currentPose = np.array(currentPose).astype(np.float64)
            # Calculate difference and error
            delta = endPose - currentPose
            error = np.linalg.norm(delta)**2
            print(f"\t [{it}] {error}")
            if error < tolerance:
                print(f"{datetime.now()}: Solution with error {error} in {it} iterations")
                return joints.ravel()

            # Compute jacobian, invert it and compute the delta in joints
            currentJacobian = self.jacobianLambda(*joints.ravel())
            currentJacobianInv = np.linalg.pinv(currentJacobian)
            deltaJoints = np.matmul(currentJacobianInv,delta)

            joints = joints + deltaJoints
            
        print(f"{datetime.now()}: Failed to compute inverse kinematics (Error: {error})")
        return init_joint_pose


    def _optimize(self, c_code:str):
        """
        Improve the output of the codegen function for better performance
        """
        import regex as re

        # Replace all divisions by their result
        matches = re.findall(r"([0-9]+)\.([0-9]+)\/([0-9]+)\.([0-9]+)", c_code)
        for elements in matches:
            string_to_be_replaced = elements[0]+'.'+elements[1]+'/'+elements[2]+'.'+elements[3]
            first_element  = float(elements[0]+'.'+elements[1])
            second_element = float(elements[2]+'.'+elements[3])
            result = first_element/second_element
            if np.isclose(result, np.pi):
                result = "M_PI"
            elif np.isclose(result, np.pi/2.0):
                result = "M_PI_2"
            c_code = c_code.replace(string_to_be_replaced,str(result))

        # Replace trigonometric identities
        for q in range(self.jointsNum):
            c_code = c_code.replace(f"sin(q[{q}] + M_PI_2)", f"cos(q[{q}])")
            c_code = c_code.replace(f"sin(q[{q}] - M_PI_2)",f"(-cos(q[{q}]))")
            c_code = c_code.replace(f"sin(q[{q}] + M_PI)", f"-sin(q[{q}])")
            c_code = c_code.replace(f"sin(q[{q}] - M_PI)", f"-sin(q[{q}])")
            c_code = c_code.replace(f"cos(q[{q}] + M_PI_2)",f"(-sin(q[{q}]))")
            c_code = c_code.replace(f"cos(q[{q}] - M_PI_2)", f"sin(q[{q}])")
            c_code = c_code.replace(f"cos(q[{q}] + M_PI)", f"-cos(q[{q}])")
            c_code = c_code.replace(f"cos(q[{q}] - M_PI)", f"-cos(q[{q}])")


        # Replace cosines and sines for precalculated values
        # TODO: find more paterns to simplify with trigonometry (like sin(x+PI), cos(x+PI), etc.)
        dict_subs = {}
        for q in range(self.jointsNum):
            dict_subs[q] = {"cos":False, "sin": False}
            if c_code.find(f"cos(q[{q}])"):
                c_code = c_code.replace(f"cos(q[{q}])", f"c{q}")
                dict_subs[q]["cos"] = True
            if c_code.find(f"sin(q[{q}])"):
                c_code = c_code.replace(f"sin(q[{q}])", f"s{q}")
                dict_subs[q]["sin"] = True

        lines = c_code.split('\n')
        for i,l in enumerate(lines):
            if '{' in l:
                for q,subs in dict_subs.items():
                    if subs["cos"]:
                        lines.insert(i+1, f"   double c{q} = cos(q[{q}]);")
                    if subs["sin"]:
                        lines.insert(i+1, f"   double s{q} = sin(q[{q}]);")
                lines[i] = l
            
        return "\n".join(lines)
    

    def genCCode(self, filename:str=None, simplify:bool = False, header:bool = False):
        """
        Write direct kinematics and jacobian matrixes into C code

        #param filename (Default=robot name): the filename where to write the .c and .h files.
        #param simplify (Default=False): the expression for dk and jacobians. It can make the process very slow.
        #param header (Default=False): generate a file for the header or not.
        """
        from sympy.utilities.codegen import codegen
        if filename is None:
            filename = self.name

        if simplify:
            self.directTransformSym = sympy.simplify(self.directTransformSym)
            self.jacobian           = sympy.simplify(self.jacobian)
            self.jacobianPos        = sympy.simplify(self.jacobianPos)
            self.jacobianOri        = sympy.simplify(self.jacobianOri)
            self.jacobianGeom       = sympy.simplify(self.jacobianGeom)

        joints_as_vector = sympy.Matrix(sympy.MatrixSymbol('q',self.jointsNum,1))
        directKin   = cp.deepcopy(self.directTransformSym)
        jacobian    = cp.deepcopy(self.jacobian)
        jacobianPos = cp.deepcopy(self.jacobianPos)
        jacobianOri = cp.deepcopy(self.jacobianOri)
        jacobianG   = cp.deepcopy(self.jacobianGeom)
        for i,q in enumerate(self.jointsSym):
            directKin    = directKin.subs(q,joints_as_vector[i])
            jacobian     = jacobian.subs(q,joints_as_vector[i])
            jacobianPos  = jacobianPos.subs(q,joints_as_vector[i])
            jacobianOri  = jacobianOri.subs(q,joints_as_vector[i])
            jacobianG    = jacobianG.subs(q,joints_as_vector[i])

        fileExpressions = [
            (f'{self.name}_DirectKin',         directKin  ),
            (f'{self.name}_Jacobian',          jacobian   ),
            (f'{self.name}_JacobianPos',       jacobianPos),
            (f'{self.name}_JacobianOri',       jacobianOri),
            (f'{self.name}_JacobianGeometric', jacobianG  )
        ]

        [(c_name, c_code), (h_name, h_code)] = codegen(fileExpressions, "C99", filename, header=False, empty=False)
        c_code = self._optimize(c_code)

        # Write C code for kinematic functions into a .c file
        with open(c_name,'w+') as c_file:
            c_file.write(c_code)

        if header:
            # Write C code for kinematic functions into a .h file
            with open(h_name,'w+') as h_file:
                h_file.write(h_code)
    
    def genURDF(self, filename:str=None, connectorLinks:bool=True):
        """
        Build a URDF file with the robot structure. Deactivate visual links in case 3D meshes have to be used. 
        """
        if filename is None:
            filename = self.name + ".urdf"
        elif not filename.endswith(".urdf"):
            filename += ".urdf"

        # Compute robot scale to determine the volume of the parts
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
            robfile.write(self._genURDFLinks(links,scale,connectorLinks))
            # Add joints
            robfile.write(self._genURDFJoints(links))
            # File ending
            robfile.write("</robot>")

    def _genURDFWorldLink(self,links:list,scale:float) -> str:
        origin = sympy.zeros(6,1)
        if self.worldToBase is not None:
            origin = Rotations().matrixToEulerRPY(self.worldToBase)
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
        string += f'\t\t<origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="{origin[3]} {origin[4]} {origin[5]}" />\n'
        string +=  '\t</joint>\n'
        return string

    def _genURDFLinks(self,links:list,scale:float,connectorLinks:bool=True) -> str:
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
            if True == connectorLinks and link > 1e-9:
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

        if self.tcpOffset is not None:
            # Add TCP
            string += f'\t<!-- ******************* TCP LINK ******************* -->\n'
            string += f'\t<link name="TCP">\n'
            # Visual section
            string +=  '\t\t<visual>\n'
            string += f'\t\t\t<origin xyz="0 0 0" rpy="0 0 0" />\n'
            string +=  '\t\t\t<geometry>\n'
            string += f'\t\t\t\t<sphere radius=\"{radius}\" />\n'
            string +=  '\t\t\t</geometry>\n'
            string += material_str
            string +=  '\t\t</visual>\n'
            # Collision section
            string +=  '\t\t<collision>\n'
            string += f'\t\t\t<origin xyz="0 0 0" rpy="0 0 0" />\n'
            string +=  '\t\t\t<geometry>\n'
            string += f'\t\t\t\t<sphere radius=\"{radius*0.95}\" />\n'
            string +=  '\t\t\t</geometry>\n'
            string +=  '\t\t</collision>\n'
            string +=  '\t</link>\n'
            string += f'\t<joint name="TCP" type="fixed">\n'
            origin = Rotations().matrixToEulerRPY(self.tcpOffset)
            string += f'\t\t<origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="{origin[3]} {origin[4]} {origin[5]}" />\n'
            string += f'\t\t<parent link="{name}_joint"/>\n'
            string += f'\t\t<child link="TCP"/>\n'
            string +=  '\t</joint>\n'

        string += '\n'
        return string

    def _genURDFJoints(self,links:list) -> str:
        string = '\t<!-- ********************* JOINTS ********************* -->\n'
        origin = [0,0,0,0,0,0]
        prev_d = 0
        prev_a = 0
        prev_alfa = 0
        prev_dr = self.denavitRows[0]
        for i,dr in enumerate(self.denavitRows):
            theta,d,a,alfa = dr.dhParams
            eulerXYZ = Rotations().matrixToEulerRPY(Rotations().rotMatrixX(prev_alfa)*Rotations().rotMatrixZ(theta))
            origin = [prev_a, 0, prev_d, eulerXYZ[0],eulerXYZ[1],eulerXYZ[2]]

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
            string += f'\t\t<origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="{origin[3]} {origin[4]} {origin[5]}" />\n'
            string += '\t</joint>\n'

            prev_d = d
            prev_a = a
            prev_alfa = alfa
            prev_dr = dr

        string += '\n'
        return string

    def _saveModel(self):
        with open(f"{self.name}.pkl",'wb') as file:
            cloudpickle.dump(self,file)
        with open(f"{self.name}.csv",'w') as file:
            file.write("theta,d,a,alpha,joint,joint_type,upper_limit,lower_limit\n")
            for dr in self.denavitRows:
                if dr.joint is not None:
                    file.write(f"{dr.dhParams[0]},{dr.dhParams[1]},{dr.dhParams[2]},{dr.dhParams[3]},{dr.joint.name},{dr.joint.type},{dr.joint.upper_limit},{dr.joint.lower_limit}\n")
                else:
                    file.write(f"{dr.dhParams[0]},{dr.dhParams[1]},{dr.dhParams[2]},{dr.dhParams[3]},,,,,\n")
    
    def  _loadModel(self):
        if os.path.exists(f"{self.name}.pkl"):
            with open(f"{self.name}.pkl",'rb') as file:
                model:DenavitDK = cloudpickle.load(file)
                # Check if the model is the same
                if model == self:
                    return model
        return None
        
class DenavitDKCsv(DenavitDK):
    def __init__(self, csvFile:str):
        denavitRows = []
        with open(csvFile) as fd:
            df = csv.reader(fd)
            # Get headers and assing indexes
            headers = next(df)
            index = dict({})
            for i,h in enumerate(headers):
                index.update({h.strip():i})
            # Build Denavit-Hartenberg table out of each row
            joints_count = 0
            for row in df:
                joint = None
                # Check if the joint name is set and assign one if not
                if "joint" in index.keys():
                    joint_name = row[index["joint"]].strip()
                    if joint_name != "":
                        joint_symbol = sympy.Symbol(joint_name)
                    else:
                        joint_symbol = sympy.Symbol(f"q{joints_count}")
                else:
                    joint_symbol = sympy.Symbol(f"q{joints_count}")
                # Check the symbol type and create the joint
                if   row[index["joint_type"]].strip().upper() == 'ROTATIONAL':
                    upper = row[index["upper_limit"]].strip() if "upper_limit" in index.keys() else  6.283185307179586
                    lower = row[index["lower_limit"]].strip() if "lower_limit" in index.keys() else -6.283185307179586
                    joint = Joint(joint_symbol, JointType.ROTATIONAL, float(upper),float(lower))
                    joints_count += 1
                elif row[index["joint_type"]].strip().upper() == 'PRISMATIC':
                    upper = row[index["upper_limit"]].strip() if "upper_limit" in index.keys() else  100000
                    lower = row[index["lower_limit"]].strip() if "lower_limit" in index.keys() else -100000
                    joint = Joint(joint_symbol, JointType.PRISMATIC, float(upper),float(lower))
                    joints_count += 1
                # Create Denavit Row and append it to the list  
                denavitRows.append(
                    DenavitRow(
                        float(row[index["theta"]]), 
                        float(row[index["d"]]), 
                        float(row[index["a"]]), 
                        float(row[index["alpha"]]), 
                        joint)
                )
        super().__init__(denavitRows,robotName=os.path.basename(csvFile).split('.')[0])


def main():
    parser = argparse.ArgumentParser(description="Kinematics Generator")
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the Denavit-Hartenberg parameters')
    parser.add_argument('--no-c', action='store_true', help='Do not generate C code for the kinematics')
    parser.add_argument('--no-urdf', action='store_true', help='Do not generate URDF file for the robot')
    args = parser.parse_args()

    if os.path.exists(args.csv_file) is False:
        raise Exception(f"File {args.csv_file} does not exist")
    
    # Load the robot from the CSV file
    print(f"{datetime.now()}: Loading robot Denavit Hartenberg table from {args.csv_file}")
    robot = DenavitDKCsv(args.csv_file)

    # Generate C code if requested
    if not args.no_c:
        print(f"{datetime.now()}: Generating C code for the robot")
        robot.genCCode()

    # Generate URDF file if requested
    if not args.no_urdf:
        print(f"{datetime.now()}: Generating URDF for the robot")
        robot.genURDF()

if __name__ == "__main__" :


    main()
    exit()

    # arm  = 10
    # farm = 5
    # palm = 1
    # fing = 1.2

    # T_shz = DenavitRow(0    , 0    , 0    , -pi/2, Joint(sympy.Symbol('sh_z'),JointType.ROTATIONAL,upper_limit=165*pi/180, lower_limit=-pi/2))
    # T_shy = DenavitRow(pi/2 , 0    , 0    , pi/2,  Joint(sympy.Symbol('sh_y'),JointType.ROTATIONAL,upper_limit=pi/2,       lower_limit=-pi/2))
    # T_shx = DenavitRow(-pi/2, -arm , 0    , pi/2,  Joint(sympy.Symbol('sh_x'),JointType.ROTATIONAL,upper_limit=pi/2,       lower_limit=-pi/2))
    # T_elz = DenavitRow(0    , 0    , 0    , -pi/2, Joint(sympy.Symbol('el_z'),JointType.ROTATIONAL,upper_limit=165*pi/180, lower_limit=0))
    # T_elx = DenavitRow(0    , -farm, 0    , pi/2,  Joint(sympy.Symbol('el_x'),JointType.ROTATIONAL,upper_limit=pi/6,       lower_limit=-110*pi/180))
    # T_wrz = DenavitRow(pi/2 , 0    , 0    , -pi/2, Joint(sympy.Symbol('wr_z'),JointType.ROTATIONAL,upper_limit=10*pi/180,  lower_limit=-pi/6))
    # T_wry = DenavitRow(0    , 0    , -palm, 0,     Joint(sympy.Symbol('wr_y'),JointType.ROTATIONAL,upper_limit=pi/3,       lower_limit=-pi/2))
    # T_hdy = DenavitRow(0    , 0    , -fing, 0,     Joint(sympy.Symbol('hd_y'),JointType.ROTATIONAL,upper_limit=5*pi/180,   lower_limit=-pi/2))

    # T_arm = DenavitDK((T_shz,T_shy,T_shx,T_elz,T_elx,T_wrz,T_wry,T_hdy),"humanArm8")
    # # T_arm.genURDF()

    # thumb = 0.8
    # T_thb = DenavitRow(0, 0, -thumb, 0)

    # T_arm5 = DenavitDK((T_shz,T_shy,T_shx,T_elz,T_elx,T_thb),"humanArm5")
    # T_arm5.genURDF()

    """
    Cartesian 6DoF
    """
    # L = 100
    # T_cartesian = DenavitDK(
    #     (
    #         DenavitRow( 0,   L,0,-pi/2,Joint(sympy.Symbol('q_x')    ,JointType.PRISMATIC)),
    #         DenavitRow(-pi/2,L,0, pi/2,Joint(sympy.Symbol('q_y')    ,JointType.PRISMATIC)),
    #         DenavitRow( 0,   L,0, 0   ,Joint(sympy.Symbol('q_z')    ,JointType.PRISMATIC)),
    #         DenavitRow( 0,   0,0,-pi/2,Joint(sympy.Symbol('q_yaw')  ,JointType.ROTATIONAL)),
    #         DenavitRow(-pi/2,0,0,-pi/2,Joint(sympy.Symbol('q_pitch'),JointType.ROTATIONAL)),
    #         DenavitRow(0    ,0,0, 0   ,Joint(sympy.Symbol('q_roll') ,JointType.ROTATIONAL)),
    #         # DenavitRow(0,   0.5,0,0),
    #     ),
    #     "orientable_cartesian",
    #     worldToBase= sympy.Matrix([
    #         [Rotations().rotMatrixY(pi/2)   , -L*sympy.ones(3,1)],
    #         [sympy.zeros(1,3)               , sympy.eye(1)]
    #     ])
    # )
    # T_cartesian.genURDF()


    """
    UR3e 
    https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    
				
    """
    T_ur3e = DenavitDK(
        (
            DenavitRow( 0, 0.15185   , 0        , pi/2  ,Joint(sympy.Symbol('q_0'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0         ,-0.24355  , 0     ,Joint(sympy.Symbol('q_1'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0         ,-0.2132   , 0     ,Joint(sympy.Symbol('q_2'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0.13105   , 0        , pi/2  ,Joint(sympy.Symbol('q_3'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0.08535   , 0        ,-pi/2  ,Joint(sympy.Symbol('q_4'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0.0921    , 0        , 0     ,Joint(sympy.Symbol('q_5'),JointType.ROTATIONAL)),
        ),
        "UR3e",
        saveModelToFile=False
    )
    # T_ur3e.genURDF(connectorLinks = False)
    T_ur3e.genCCode()

    print(T_ur3e.eval((0, 0, 0, 0, 0, 0)))
    endpose = np.array((
        # ( 1, 0, 0,-0.45675),
        # ( 0, 0,-1,-0.22315),
        # ( 0, 0,-1,-0.0),
        # ( 0, 1, 0, 0.0665),
        # ( 0, 0, 0, 1)
        ( 1, 0, 0,-0.30),
        ( 0, 0,-1,-0.10),
        ( 0, 1, 0, 0.0665),
        ( 0, 0, 0, 1)
    ))

    print(T_ur3e.inverseEval((0,0,0,0,0,0),endpose, tolerance=1e-5))
