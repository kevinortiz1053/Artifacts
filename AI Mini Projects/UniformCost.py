#THE PANCAKE PROBLEM
#Uniform-cost Search
import numpy as np
FAILURE = 0


def initial_stack():
    x = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = x.tolist()
    return y

#class to count number of gaps
#returns the count


class GapCounter:
    def run(self):
        count = 0
        for i in range(len(self)-1):
            if abs(self[i] - self[i+1]) > 1:
                count = count + 1
        return count


class UniformCost:
    def run(self):
        frontier = [self.copy()]
        cost = [0]
        goal = self.copy()
        goal.sort()
        goal1 = goal.copy()
        goal1.reverse()
        visited = []
        while self != goal:
            if len(frontier) == 0:
                return FAILURE
            ind = cost.index(min(cost))
            expansion = []
            for i in range(len(self)-2):  # expanding parent node
                b = self.copy()
                x = b[0:i+2]
                x.reverse()
                y = b[i+2:]
                z = x + y
                expansion.append(z)
            b.reverse()
            # expanding last child which is just a reverse of the parent
            expansion.append(b)
            for j in expansion:
                if j not in frontier and j not in visited:
                    frontier.append(j)
                    cost.append(cost[ind] + GapCounter.run(j))
                if j in frontier:
                    ind1 = frontier.index(j)
                    c = cost[ind1]
                    g1 = cost[ind] + GapCounter.run(j)
                    if g1 < c:
                        cost[ind1] = g1
            cost.pop(ind)
            ind2 = cost.index(min(cost))
            visited.append(self)
            frontier.pop(ind2)
            self = frontier[ind2]
            if self == goal1:
                self.reverse()
            if self == goal:
                prize = self.copy()
                return prize


stack = initial_stack()
#stack = [2,1,4,10,3,6,5,7,9,8]
print(stack)
solution = UniformCost.run(stack)
print("Solution:", solution)
