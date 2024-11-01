import numpy as np
from typing import List, Tuple

MARKER_SIZE = 110
DELTA = 30

class Position(object):
    """
    Represents a position in a 2D plane.
    """
    def __init__(self, delta: float, angle: float, id: int = 0):
        super(Position, self).__init__()
        self.id = id
        self.delta = delta
        self.rad = angle
        self.deg = np.rad2deg(angle)
        self.x = delta*np.cos(angle)
        self.y = delta*np.sin(angle)

class Cell(object):
    """
    Represents a cell in a grid.
    """
    def __init__(self, size: int, row: int, col: int, zone: 'Zone' = None, free=1):
        super(Cell, self).__init__()
        self.size = size
        self.free = free
        self.row = row
        self.col = col
        self.zone = zone

    def __str__(self) -> str:
        if self.zone:
            return "{0}[zone: ({1}, {2}), row: {3}, col: {4}, cx: {5}, cy: {6}]"\
            .format(self.__class__.__qualname__, self.zone.row, self.zone.col, self.row, self.col, self.cx, self.cy)
        return "{0}[row: {1}, col: {2}, cx: {3}, cy: {4}]"\
                .format(self.__class__.__qualname__, self.row, self.col, self.cx, self.cy)

    @property
    def cx(self) -> float:
        if self.zone:
            return self.zone.col*self.zone.size + self.col*self.size + self.size/2 
        return self.col*self.size + self.size/2
    
    @property
    def cy(self) -> float:
        if self.zone:
            return self.zone.row*self.zone.size + self.row*self.size + self.size/2
        return self.row*self.size + self.size/2
    
class Marker(Cell):
    def __init__(self, size: int, centroid: Cell, id: int):
        super(Marker, self).__init__(size, centroid.row, centroid.col, centroid.zone, 0)
        self.__centroid__ = centroid
        self.id = id

    @property
    def cx(self) -> float:
        return self.__centroid__.cx
    
    @property
    def cy(self) -> float:
        return self.__centroid__.cy

class Zone(Cell):
    def __init__(self, size: int, row: int, col: int, cells=1):
        super(Zone, self).__init__(size, row, col)
        self.size = size
        self.cell_size = size//cells
        self.cells: List[List[Cell]] = []
        self.markers: List[Marker] = []

        for row in range(cells):
            self.cells.append([])
            for col in range(cells):
                self.cells[row].append(Cell(size/cells, row, col, self))

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, rxc: tuple) -> Cell:
        return self.cells[rxc[0]][rxc[1]]

    def diffuse(self, cells: int = 9):
        self.cell_size = self.size//cells

        n = len(self)
        for row in range(cells):
            if row >= n:
                self.cells.append([])
            for col in range(cells):
                if row >= n or col >= n:
                    self.cells[row].append(Cell(self.cell_size, row, col, self))
                else:
                    self.cells[row][col].size = self.cell_size
        return self

class Grid(object):
    def __init__(self, origo: Tuple[int, int], zone_size: int, zones=9, marker_size=110):
        super(Grid, self).__init__()
        self.zone_size = zone_size
        self.marker_size = marker_size
        self.zones: List[List[Zone]] = []
        self.markers: List[Marker] = []

        for row in range(zones):
            self.zones.append([])
            for col in range(zones):
                self.zones[row].append(Zone(zone_size, row, col))
        self.origo = self[origo][0, 0]

    def __len__(self):
        return len(self.zones)

    def __getitem__(self, rxc: tuple) -> Zone:
        return self.zones[rxc[0]][rxc[1]]
    
    def transform_xy(self, x: float, y: float):
        dx, dy = max(1, min(x, self.zone_size*(len(self)-1))), max(1, min(y, self.zone_size*(len(self)-1)))
        row, col = int(dy//self.zone_size), int(dx//self.zone_size)
        zone = self[row, col]
        if len(zone.cells) == 1:
            return zone[0, 0]
 
        sx, sy = 1-col*zone.size/dx, 1-row*zone.size/dy
        dx, dy = dx*sx, dy*sy
        row, col = int(dy//zone.cell_size), int(dx//zone.cell_size)
        return zone[row, col]
    
    def transform_position(self, pos: Position):
        return self.transform_xy(pos.x, pos.y)

    def transform_pose(self, pose: Position):
        dx, dy = pose.x + self.origo.cx, pose.y + self.origo.cy
        return self.transform_xy(
            self.origo.cx + pose.delta*np.cos(pose.rad), 
            self.origo.cy + pose.delta*np.sin(pose.rad)
        )

    def transform_cell(self, cell: Cell):
        return Position(
            np.sqrt(cell.cx**2 + cell.cy**2), np.arctan2(cell.cy, cell.cx)
        )
    
    def create_marker(self, zone: Zone, cell: Cell, id, size=MARKER_SIZE, delta=DELTA):
        # https://en.wikipedia.org/wiki/Circular_segment
        offset = int((delta/2)+(size**2/(8*delta)))
        # cell is the edge of the of the marker, so get the centroid
        dx, dy = zone.cx - cell.cx, zone.cy - cell.cy
        theta = np.arctan2(dy, dx)
        delta = np.sqrt(2*(size/2)**2)
        centroid = self.transform_xy(cell.cx + delta*np.cos(theta), cell.cy + delta*np.sin(theta))
        m = Marker(offset*2, centroid, id)

        for ky in range(-offset, offset, zone.cell_size):
            for kx in range(-offset, offset, zone.cell_size):
                dx, dy = centroid.cx+kx, centroid.cy+ky

                if not (0 <= dx < len(self)*self.zone_size) or not (0 <= dy < len(self)*self.zone_size):
                    continue

                cell = self.transform_xy(dx, dy)
                if cell.zone.free:
                    cell.zone.diffuse().free = 0
                    cell.zone.markers.append(m)
                    self.markers.append(m)
                    cell = self.transform_xy(dx, dy)
                cell.free = 0

    def nearest_marker(self, cell: Cell):
        min = np.inf
        nearest = self.origo
        for o in self.markers:
            d = np.sqrt((o.cx-cell.cx)**2+(o.cy-cell.cy)**2)
            if d < min:
                min = d
                nearest = o
        return nearest
    
    def update(self, origo: Cell, pose: Position=None, id=0):
        self.origo = origo

        if pose is None:
            return
        
        cell = self.transform_pose(pose)
        if cell.zone.free:
            if cell.free:
                self.create_marker(cell.zone.diffuse(), self.transform_pose(pose), id)
        elif cell.free:
            self.create_marker(cell.zone, cell, id)

    def random_cell(self):
        row, col = np.random.randint(0, len(self)), np.random.randint(0, len(self))
        if self[row, col].free:
            return self[row, col][0, 0]
        
        crow, ccol = np.random.randint(0, len(self)), np.random.randint(0, len(self))
        return self[row, col][crow, ccol]

def main():
    grid = Grid((4, 4), 450)

    grid[1, 1].diffuse()
    assert(len(grid[1, 1]) == 9)
    assert(grid[1, 1].cell_size == grid[0, 0].size//9)
    assert(grid[1, 1].cx == 450*1.5)
    assert(grid[1, 1].cy == 450*1.5)

    grid.create_marker(grid[1, 1], grid[1, 1][0, 0], 0)
    assert(grid[1, 1][1, 1].free == 0)

    p = grid.transform_cell(grid[5, 5])
    assert(grid[5, 5].cx-0.0001 < p.x < grid[5, 5].cx+0.0001)
    assert(grid[5, 5].cy-0.0001 < p.x < grid[5, 5].cy+0.0001)

    c = grid.transform_pose(Position(np.sqrt(2*450**2), 0.25*np.pi))
    assert(c.zone.row == 5 and c.zone.col == 5)

if __name__ == '__main__':
    main()
    print(__file__, "\033[32m✓\033[0m")
