import sys
import os
os.environ['NO_BPY']='1'
from pathlib import Path

from SourceIO.source1.mdl import SourceModel

from SourceIO.source1.mdl.qc_generator import QC

if __name__ == '__main__':
    mdl_name = Path(sys.argv[1]).absolute()
    if len(sys.argv) > 2:
        out_path = Path(sys.argv[-1]) / mdl_name.stem
    else:
        out_path = Path.cwd() / 'DECOMPILED' / mdl_name.stem
    smodel = SourceModel(mdl_name)
    smodel.read()
    qc = QC(smodel)
    qc.write_qc(out_path)
    qc.smd.write_meshes(out_path)
