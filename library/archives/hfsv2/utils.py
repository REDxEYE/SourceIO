def calculate_header_offset(filename: str):
    offset = 0
    for char in filename.lower():
        offset += ord(char)
    return offset % 312 + 30


def calculate_entry_table_offset(filename: str):
    offset = 0
    for char in filename.lower():
        offset += ord(char) * 3
    return offset % 212 + 33
