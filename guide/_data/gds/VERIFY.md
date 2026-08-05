# GDS demo files — parser verification

Loaded with the app's own `tools.process.ebeam.gdsii.parser._parse_gds` (the function `gdsii/stream.py` itself calls to decode an upload).

### first_exposure.gds

unit = 1.000e-06 m/user-unit (OK 1 um)
cells = ['TOP']

cell `TOP`:
  - layer 1, datatype 0: 4 polygons, bbox (um) = (80.0, 80.0) - (1920.0, 1920.0)
  - layer 1, datatype 1: 6 polygons, bbox (um) = (730.0, 820.0) - (1270.0, 1180.0)
  - layer 1, datatype 2: 6 polygons, bbox (um) = (797.5, 845.0) - (1302.5, 1155.0)
  - layer 1, datatype 3: 6 polygons, bbox (um) = (830.0, 820.0) - (1370.0, 1180.0)

expected per-(layer,datatype) polygon counts: {(1, 1): 6, (1, 2): 6, (1, 3): 6, (1, 0): 4}
got:                                          {(1, 0): 4, (1, 1): 6, (1, 2): 6, (1, 3): 6}
MATCH: True

### second_exposure.gds

unit = 1.000e-06 m/user-unit (OK 1 um)
cells = ['TOP']

cell `TOP`:
  - layer 1, datatype 0: 4 polygons, bbox (um) = (80.0, 80.0) - (1920.0, 1920.0)
  - layer 2, datatype 1: 6 polygons, bbox (um) = (730.0, 885.0) - (1370.0, 1195.0)
  - layer 2, datatype 2: 12 polygons, bbox (um) = (725.0, 885.0) - (1375.0, 1195.0)

expected per-(layer,datatype) polygon counts: {(1, 0): 4, (2, 1): 6, (2, 2): 12}
got:                                          {(1, 0): 4, (2, 1): 6, (2, 2): 12}
MATCH: True

### dose_test.gds

unit = 1.000e-06 m/user-unit (OK 1 um)
cells = ['TOP']

cell `TOP`:
  - layer 1, datatype 0: 1 polygons, bbox (um) = (1880.0, 80.0) - (1920.0, 120.0)
  - layer 1, datatype 1: 25 polygons, bbox (um) = (1301.0, 201.0) - (1469.0, 369.0)
  - layer 1, datatype 2: 75 polygons, bbox (um) = (1311.0, 201.0) - (1477.5, 379.0)

expected per-(layer,datatype) polygon counts: {(1, 1): 25, (1, 2): 75, (1, 0): 1}
got:                                          {(1, 1): 25, (1, 2): 75, (1, 0): 1}
MATCH: True


**Overall PASS: True**
