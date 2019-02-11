import math

def convert_rotation_matrix_to_degrees(m0, m1, m2, m3, m4, m5, m8):
    angleY = -math.asin(round(m2,6))
    c = math.cos(angleY)
    if abs(c) > 0.005:
        translateX = m8/c
        translateY = -m5/c
        angleX = (math.atan2(translateY,translateX))
        translateX = m0 / c
        translateY = -m1 / c
        angleZ = (math.atan2(translateY,translateX))
    else:
        angleX = 0
        translateX = m4
        translateY = m3
        angleZ = (math.atan2(translateY, translateX))
    return angleX,angleY,angleZ