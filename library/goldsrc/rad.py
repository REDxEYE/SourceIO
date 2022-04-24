def parse_rad(rad_file):
    light_info = {}
    for line in rad_file.readlines():
        line = line.decode('utf-8').strip('\n\t\r')
        if not line or line.startswith('//'):
            continue
        mat_name, *light_data = line.split()
        light_data = [float(c) for c in light_data[:4]]
        r, g, b, l1 = convert_light_value(light_data)
        print(mat_name)
        light_info[mat_name.upper()] = [r, g, b, l1]
    return light_info


def convert_light_value(light_data):
    if len(light_data) == 4:
        r, b, g, radius = light_data
    elif len(light_data) == 1:
        r = g = b = light_data
        radius = 1
    else:
        r, b, g = light_data
        radius = 1
    l1 = max((r, g, b)) / 255
    l1 *= radius
    l1 = (l1 ** 2) / 10

    return r / 255, g / 255, b / 255, l1 / 100
