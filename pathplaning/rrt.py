import numpy as np
import matplotlib.pyplot as plt
from  pathplaning.grid import Grid, Cell

class Node(object):
    def __init__(self, cell: Cell, link: "Node" = None) -> None:
        super(Node, self).__init__()
        self.cell = cell
        self.left = None
        self.right = None
        self.link = link

class Tree(object):
    def __init__(self, root: Node):
        super(Tree, self).__init__()
        self.root = root

    def insert(self, new_node: Node):
        if not self.root:
            self.root = new_node
            return
        
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            if not node.left:
                node.left = new_node
                return
            if not node.right:
                node.right = new_node
                return
            stack.append(node.left)
            stack.append(node.right)
            
    def search(self, cell: Cell):
        if not self.root:
            return None
        
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            if node is None:
                continue
            if node.cell is cell:
                return node
            stack.append(node.left)
            stack.append(node.right)

    def nearest(self, cell: Cell):
        if not self.root:
            return None
        
        stack = [self.root]
        nearest = self.root
        min = np.inf
        while len(stack):
            node = stack.pop()
            if node is None:
                continue

            dx, dy = node.cell.cx - cell.cx, node.cell.cy - cell.cy
            dd = np.sqrt((dx)**2 + (dy)**2)
            if cell is not nearest.cell and dd < min:
                nearest, min = node, dd
            stack.append(node.left)
            stack.append(node.right)
        return nearest
    
def collision(grid: Grid, start: Cell, goal: Cell, ax=None):
    dx, dy = goal.cx - start.cx, goal.cy - start.cy
    delta = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    offsets = ((-1, -1, 0.25*np.pi), (-1, 1, -0.25*np.pi))
    for m in grid.markers:
        for kx, ky, ktheta in offsets:
            # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
            mlx, mly = m.cx+kx*grid.zone_size/2, m.cy+ky*grid.zone_size/2
            # mrx, mry = m.cx-kx*grid.zone_size/2, m.cy-ky*grid.zone_size/2
            r = np.array([delta*np.cos(theta), delta*np.sin(theta)])
            s = np.array([np.sqrt(2*m.size**2)*np.cos(ktheta), np.sqrt(2*m.size**2)*np.sin(ktheta)])
            qdelta = np.array([mlx-start.cx, mly-start.cy])


            rxs = np.cross(r, s)
            eps = 1e-9

            if -eps <= rxs <= eps:
                return True
            
            t = np.cross(qdelta, s)/rxs
            u = np.cross(qdelta, r)/rxs
            if rxs != 0 and 0 <= t <= 1 and 0 <= u <= 1:
                return True
            
    return False


    
                

def rrt_path(grid: Grid, start: Cell, goal: Cell, delta: float, n=100, ax=None):
    tree = Tree(Node(start))
    lines = []

    for i in range(n):
        if i % (n//4) == 0:
            cell = goal
        else:
            cell = grid.random_cell()
        nearest = tree.nearest(cell)

        if cell is nearest.cell:
            continue

        if not collision(grid, nearest.cell, cell, ax):
            dx, dy = cell.cx-nearest.cell.cx, cell.cy-nearest.cell.cy
            theta = np.arctan2(dy, dx)
            dx, dy = nearest.cell.cx+delta*np.cos(theta), nearest.cell.cy+delta*np.sin(theta)
            new = grid.transform_xy(dx, dy)
            if not new.free:
                continue 
            new = Node(new, nearest)
            tree.insert(new)

            if ax:
                line, = ax.plot([nearest.cell.cx, new.cell.cx], [nearest.cell.cy, new.cell.cy], 'g-', alpha=0.6)
                m, = ax.plot(new.cell.cx, new.cell.cy, 'bo', alpha=0.6)
                lines.append(line)
                lines.append(m)
                plt.draw()
                plt.pause(0.1)

            if cell is goal:
                break

    for line in lines:
            line.remove()

    if ax:
        plt.draw()
        plt.pause(0.1)

    path = [grid.transform_cell(goal)]
    node = tree.nearest(goal)
    while node:
        path.append(grid.transform_cell(node.cell))
        node = node.link

    return path

def main():
    grid = Grid((0, 0), 450)
    grid.create_marker(grid[4, 1].diffuse(), grid[4, 1][4, 4], 1)
    grid.create_marker(grid[1, 4].diffuse(), grid[1, 4][4, 4], 2)
    grid.create_marker(grid[7, 4].diffuse(), grid[7, 4][4, 4], 3)
    grid.create_marker(grid[4, 7].diffuse(), grid[4, 7][4, 4], 4)
    grid.create_marker(grid[4, 4].diffuse(), grid[4, 4][0, 0], 5)

    assert(collision(grid, grid[5, 7][0, 0], grid[3, 7][0, 0]))

    for _ in range(1000):
        rrtpath = rrt_path(grid, grid[0, 0][0, 0], grid[8, 8][0, 0], 450)
        while(len(rrtpath)):
            cell = grid.transform_position(rrtpath.pop())
            for r in range(len(grid)):
                for c in range(len(grid)):
                    for m in grid[r, c].markers:
                        assert(cell is not m)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_title('RTT')
    ax.set_xlim(0, 9**2*50)
    ax.set_ylim(0, 9**2*50)
    main()
    print(__file__, "\033[32m✓\033[0m")
