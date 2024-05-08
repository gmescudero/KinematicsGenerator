import unittest as ut # https://docs.python.org/3/library/unittest.html
import sympy
from cmath import pi
from kinematicBuilder import Rotations
import random

def deg2rad(d:float) -> float:
    return d*(pi/180)

class TestRotations(ut.TestCase):
    

    def test_matrixToEulerRPY_singularm90(self):
        """Check a conversion to EULER rpy with -90° in y"""
        
        rotM   = Rotations().rotMatrixZ(-pi)*Rotations().rotMatrixY(-pi/2)
        result = Rotations().matrixToEulerRPY(rotM)
        # print(rotM)
        for dist,expected in zip(result,[-pi,-pi/2,0]):
            self.assertAlmostEqual(dist, expected)


    def test_matrixToEulerRPY_singular90(self):
        """Check a conversion to EULER rpy with 90° in y"""
        rotM   = Rotations().rotMatrixZ(-pi)*Rotations().rotMatrixY(pi/2)
        result = Rotations().matrixToEulerRPY(rotM)
        # print(rotM)
        for dist,expected in zip(result,[pi,pi/2,0]):
            self.assertAlmostEqual(dist, expected)
    

    def test_matrixToEulerRPY_noSingular(self):
        """Test matrix conversion to Roll Pitch Yaw in a matrix out of the Euler singularity"""
        cases = [
            [sympy.Matrix([
                [ 0.0000000,  0.9659258, -0.2588190],
                [-0.5000000, -0.2241439, -0.8365163],
                [-0.8660254,  0.1294095,  0.4829629]]), 
             [deg2rad(15), deg2rad(60), deg2rad(-90)]
            ]
        ]
        for rotM,euler in cases:
            result = Rotations().matrixToEulerRPY(rotM)
            for dist,expected in zip(result,euler):
                self.assertAlmostEqual(dist, expected)



if __name__ == '__main__':
    ut.main()