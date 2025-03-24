import numpy as np
import sympy
from numpy import pi

def compute_error(current, target):
    error = np.linalg.norm(target.flatten()-current.flatten())**2
    return error

def prismatics_solve(forward_kin, jacobian, prismatics_map, target, joints):
    for j,isPrismatic in enumerate(prismatics_map):
        if isPrismatic:
            joints[j] = 0.0
    
    currentJ = jacobian(*joints)[0:3,:]

    for j,isPrismatic in enumerate(prismatics_map):
        if isPrismatic:
            joint_dir = currentJ[:,j]
            joint_dir = joint_dir / np.linalg.norm(joint_dir)

            current = forward_kin(*joints).flatten()
            target_dir = target[0:3] - current[0:3]

            joints[j] = np.dot(target_dir, joint_dir)
    return joints

def clamp_delta(delta, maxDeltas):
    for i,d in enumerate(delta):
        if d > maxDeltas[i]:
            delta[i] = maxDeltas[i]
        if d < -maxDeltas[i]:
            delta[i] = -maxDeltas[i]
    return delta


def jacobian_transposed(forward_kin, jacobian, jointsNum, target_in, joints=None, max_iter=150, epsilon=1e-6):
    """
    Inverse kinematics algorithm using the transposed jacobian method.
    """
    if target_in.shape[0] == 3:
        target = np.concat([target_in[0:3,3], target_in[0:3,0], target_in[0:3,1] ])
    else:
        target = target_in.flatten()
    if joints is None:
        joints = np.zeros(jointsNum)

    for i in range(max_iter):
        # Compute error
        current = forward_kin(*joints).flatten()
        error = compute_error(current, target)
        if error < epsilon:
            print(f"\t{i} - e:{error} - DONE")
            break
        if (i%10 == 0): print(f"\t{i} - e:{error}")

        # Iterate
        current = forward_kin(*joints).flatten()
        deltaParams = target - current

        currentJ = jacobian(*joints)
        jT = currentJ.T
        upper = currentJ @ jT @ deltaParams
        alfa = (np.dot(deltaParams, upper) / np.dot(upper, upper)) 
        joints = joints + (alfa * jT) @ deltaParams


    print(f"\t{i} - e:{error} - FINAL")
    return joints

def jacobian_dls(forward_kin, jacobian, jointsNum, target_in, joints=None, max_iter=100, epsilon=1e-6, damp_factor=0.01):
    """
    Inverse kinematics algorithm using the damped least squares jacobian method.
    """
    if target_in.shape[0] == 3:
        target = np.concat([target_in[0:3,3], target_in[0:3,0], target_in[0:3,1] ])
    else:
        target = target_in.flatten()
    if joints is None:
        joints = np.zeros(jointsNum)
    error = 1e200
    oldError = 0
    for i in range(max_iter):
        current = forward_kin(*joints).flatten()
        delta = target - current
        oldError = error
        error = np.linalg.norm(delta)
        if error < epsilon:
            print(f"\t{i} - e:{error} - DONE")
            break
        elif (error > oldError) and damp_factor < 10: 
            damp_factor = damp_factor * 1.5
            print(f"\t{i} - New damp factor: {damp_factor}")
        elif (oldError-error < epsilon): 
            damp_factor = damp_factor * 0.5
            print(f"\t{i} - New damp factor: {damp_factor}")

        if (i%10 == 0): print(f"\t{i} - e:{error} ")

        currentJ = jacobian(*joints)
        jT = currentJ.T
        jInv = np.linalg.inv(jT @ currentJ + damp_factor * np.eye(currentJ.shape[1])) @ jT
        # Update joints
        joints = joints + jInv @ delta

    print(f"\t{i} - e:{error} - FINAL")
    return joints




if __name__ == "__main__" :
    # example ur3
    from kinematicBuilder import DenavitDK, DenavitRow, Joint, JointType
    """
    UR3e 
    https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    
    """
    T_ur3e = DenavitDK(
        (
            DenavitRow( 0, 0.15185, 0        , pi/2  ,Joint(sympy.Symbol('q_0'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0      ,-0.24355  , 0     ,Joint(sympy.Symbol('q_1'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0      ,-0.2132   , 0     ,Joint(sympy.Symbol('q_2'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0.13105, 0        , pi/2  ,Joint(sympy.Symbol('q_3'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0.08535, 0        ,-pi/2  ,Joint(sympy.Symbol('q_4'),JointType.ROTATIONAL)),
            DenavitRow( 0, 0.0921 , 0        , 0     ,Joint(sympy.Symbol('q_5'),JointType.ROTATIONAL)),
        ),
        "UR3e"
    )

    # Target pose as a 4x4 matrix
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
    print("Test the jacobian transposed method for Inverse Kinematics")
    print(jacobian_transposed(T_ur3e.directLambdaTransform, T_ur3e.jacobianLambda, 6, endpose))
    print("Test the jacobian damped least squares method for Inverse Kinematics")
    print(jacobian_dls(T_ur3e.directLambdaTransform, T_ur3e.jacobianLambda, 6, endpose))