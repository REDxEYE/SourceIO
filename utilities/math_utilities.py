import math

from SourceIO.data_structures.source_shared import SourceVector


def convert_rotation_matrix_to_degrees(m0, m1, m2, m3, m4, m5, m8):
    angle_y = -math.asin(round(m2, 6))
    c = math.cos(angle_y)
    if abs(c) > 0.005:
        translate_x = m8 / c
        translate_y = -m5 / c
        angle_x = (math.atan2(translate_y, translate_x))
        translate_x = m0 / c
        translate_y = -m1 / c
        angleZ = (math.atan2(translate_y, translate_x))
    else:
        angle_x = 0
        translate_x = m4
        translate_y = m3
        angleZ = (math.atan2(translate_y, translate_x))
    return angle_x, angle_y, angleZ


#	Public Function VectorITransform(ByVal input As SourceVector,
#  ByVal matrixColumn0 As SourceVector,
#  ByVal matrixColumn1 As SourceVector,
#  ByVal matrixColumn2 As SourceVector,
#  ByVal matrixColumn3 As SourceVector) As SourceVector
#		Dim output As SourceVector
#		Dim temp As SourceVector
#
#		output = New SourceVector()
#		temp = New SourceVector()
#
#		temp.x = input.x - matrixColumn3.x
#		temp.y = input.y - matrixColumn3.y
#		temp.z = input.z - matrixColumn3.z
#
#		output.x = temp.x * matrixColumn0.x + temp.y * matrixColumn0.y + temp.z * matrixColumn0.z
#		output.y = temp.x * matrixColumn1.x + temp.y * matrixColumn1.y + temp.z * matrixColumn1.z
#		output.z = temp.x * matrixColumn2.x + temp.y * matrixColumn2.y + temp.z * matrixColumn2.z
#
#		Return output
#	End Function
def vector_i_transform(input: SourceVector,
                       matrix_c0: SourceVector, 
                       matrix_c1: SourceVector,
                       matrix_c2: SourceVector,
                       matrix_c3: SourceVector):
    temp = SourceVector()
    output = SourceVector()

    temp.x = input.x - matrix_c3.x
    temp.y = input.y - matrix_c3.y
    temp.z = input.z - matrix_c3.z

    output.x = temp.x * matrix_c0.x + temp.y * matrix_c0.y + temp.z * matrix_c0.z
    output.y = temp.x * matrix_c1.x + temp.y * matrix_c1.y + temp.z * matrix_c1.z
    output.z = temp.x * matrix_c2.x + temp.y * matrix_c2.y + temp.z * matrix_c2.z

    return output
