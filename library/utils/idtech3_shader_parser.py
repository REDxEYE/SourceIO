import hashlib

CACHE = {}


def parse_shader_materials(shader_data: str):
    hash_key = hashlib.md5(shader_data.encode("utf8")).hexdigest()
    if hash_key in CACHE:
        return CACHE[hash_key]
    lines = []
    shader_data = shader_data.replace("\t", " ")
    shader_data = shader_data.replace("\r\n", "\n")
    shader_data = shader_data.replace("\r", "")
    for line in shader_data.split(" "):
        while '\n' in line:
            slit_point = line.index("\n")
            lines.append(line[:slit_point])
            lines.append("\n")
            line = line[slit_point + 1:]
        else:
            lines.append(line)
    lines = list(filter(lambda a:a!="", lines))

    def skip_comments_and_whitespace():
        while lines:
            if lines[0].isspace() or not lines[0]:
                lines.pop(0)
                continue
            if lines and lines[0].startswith("//"):
                while lines[0] != "\n":
                    lines.pop(0)
                lines.pop(0)
                continue
            break

    def get_next():
        skip_comments_and_whitespace()
        return lines.pop(0)

    def get_until(terminator: str):
        parts = []
        while not match(terminator, True):
            parts.append(lines.pop(0))
        return " ".join(parts)

    def peek_next():
        skip_comments_and_whitespace()
        return lines[0]

    def must(key: str, consume: bool = True) -> bool:
        if peek_next() != key:
            raise SyntaxError(f"Unexpected key, expected {key}, but got {peek_next()}")
        if consume:
            get_next()
        return True

    def match(key: str, consume: bool = True) -> bool:
        if key.isspace():
            peek = lines[0]
        else:
            peek = peek_next()
        if peek == key:
            if consume:
                lines.pop(0)
            return True
        return False

    def parse_block(mat_def: bool = True):
        material = {}
        if mat_def:
            material["textures"] = []
        while not match("}", True):
            if match("{", True):
                material["textures"].append(parse_block(False))
                continue

            key = get_next()
            value = get_until("\n")
            material[key.lower()] = value
        return material

    materials = {}
    skip_comments_and_whitespace()
    if lines:
        while lines:
            mat_name = get_next()
            must("{", True)

            materials[mat_name] = parse_block(True)
            skip_comments_and_whitespace()
    CACHE[hash_key] = materials
    return materials
