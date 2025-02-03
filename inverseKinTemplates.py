import sympy
from kinematicBuilder import Rotations

class TwoLinkPlanar:
    """
    Kinematic definition of a two link planar robot (2DOF)
    """

    def __init__(self, a1=None,a2=None,q1=None,q2=None,x=None,y=None,theta=None) -> None:
        self.a1 = sympy.Symbol('a_1') if a1 is None else a1
        self.a2 = sympy.Symbol('a_2') if a2 is None else a2
        self.q1 = sympy.Symbol('q_1') if q1 is None else q1
        self.q2 = sympy.Symbol('q_2') if q2 is None else q2
        self.x  = sympy.Symbol('x')   if x is None else x
        self.y  = sympy.Symbol('y')   if y is None else y
        self.theta = sympy.Symbol('theta') if theta is None else theta
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
    """
    Kinematic definition of a two link robot mounted on a rotatory base (3DOF)
    """
    def __init__(self, a1=None,a2=None,q1=None,q2=None,q3=None,x=None,y=None,z=None,r_expr=None) -> None:
        self.a1 = sympy.Symbol('a_1') if a1 is None else a1
        self.a2 = sympy.Symbol('a_2') if a2 is None else a2
        self.q1 = sympy.Symbol('q_1') if q1 is None else q1
        self.q2 = sympy.Symbol('q_2') if q2 is None else q2
        self.q3 = sympy.Symbol('q_3') if q3 is None else q3
        self.x = sympy.Symbol('x') if x is None else x
        self.y = sympy.Symbol('y') if y is None else y
        self.z = sympy.Symbol('z') if z is None else z
        self._r = sympy.Symbol('r')
        self.r_expr = sympy.sqrt(self.x**2 + self.y**2) if r_expr is None else r_expr
        self.twoLink = TwoLinkPlanar(self.a1,self.a2,self.q2,self.q3,self._r,self.z)
        # Compute direct and inverse with default symbols
        self._directSymBase  = self.directKinHomoTSymbols()
        self._inverseSymBase = self.inverseKinSymbols()
        # Compute direct and inverse kinematics with custom symbols
        self.setSymbols(a1,a2,q1,q2,q3,x,y,z,r_expr) 


    def setSymbols(self,a1=None,a2=None,q1=None,q2=None,q3=None,x=None,y=None,z=None,r_expr=None):
        self._substitutions = {
            self.a1:a1 if a1 is not None else self.a1,
            self.a2:a2 if a2 is not None else self.a2,
            self.q1:q1 if q1 is not None else self.q1,
            self.q2:q2 if q2 is not None else self.q2,
            self.q3:q3 if q3 is not None else self.q3,
            self.x:x if x is not None else self.x,
            self.y:y if y is not None else self.y,
            self.z:z if z is not None else self.z,
        }
        if r_expr is None:
            self.r_expr = sympy.sqrt(self.x**2 + self.y**2).subs(self._substitutions,simultaneous=True)
        else:
            self.r_expr = r_expr.subs(self._substitutions,simultaneous=True)
        self._substitutions.update({self._r:self.r_expr})

        self.twoLink.setSymbols(self._substitutions[self.a1], self._substitutions[self.a2], 
                                self._substitutions[self.q2], self._substitutions[self.q3],
                                self.r_expr, self._substitutions[self.z] )

        self.directSym  = self._directSymBase.subs(self._substitutions,simultaneous=True).subs(self._r,self.r_expr)
        self.inverseSym = self._inverseSymBase.subs(self._substitutions,simultaneous=True).subs(self._r,self.r_expr)
            
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
    """
    Kinematic definition of a spherical wrist consisting of a sequence of 3 rotations whose axes meet at a specific 
    point (3DOF)
    """
    def __init__(self, sequence:str, q1=None,q2=None,q3=None,yaw=None,pitch=None,roll=None) -> None:
        self.q1    = sympy.Symbol('q_1') if q1 is None else q1
        self.q2    = sympy.Symbol('q_2') if q2 is None else q2
        self.q3    = sympy.Symbol('q_3') if q3 is None else q3
        self.yaw   = sympy.Symbol('gamma') if yaw is None else yaw
        self.pitch = sympy.Symbol('beta')  if pitch is None else pitch
        self.roll  = sympy.Symbol('alpha') if roll is None else roll
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
    """
    Kinematic definiton of a decapled robot arm consisting of a two link planar mounted on a rotatory base and an 
    spherical wrist mounted at the end of it, so that the position of the control point can be determined by the first
    part and the orientation can be adjusted by the second (6DOF)
    """
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
