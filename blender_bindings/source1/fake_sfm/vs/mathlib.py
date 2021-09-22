def VectorSubtract(pos_b, pos_a, dir):
    dir.x = pos_b.x - pos_a.x
    dir.y = pos_b.y - pos_a.y
    dir.z = pos_b.z - pos_a.z


def VectorScale(vec, scale, res):
    res.x = vec.x * scale #* 1 / 20
    res.y = vec.y * scale #* 1 / 20
    res.z = vec.z * scale #* 1 / 20
