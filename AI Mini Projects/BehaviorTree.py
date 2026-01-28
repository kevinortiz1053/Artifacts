from typing import List, Dict, Any
import string

FAILED = 0
SUCCEEDED = 1
RUNNING = 2


class Node:
    _name: int

    def __init__(self, name: int):
        self._name = name

    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        return FAILED


class Composite(Node):
    _children: List[Node]

    def __init__(self, name: int, children: List[Node]):
        super().__init__(name)
        self._children = children


class Decorator(Node):
    _child: Node

    def __init__(self, name: int, child: Node):
        super().__init__(name)
        self._child = child


class Task(Node):
    pass


class Condition(Node):
    pass

# Composites definitions


class Sequence(Composite):
    def __init__(self, name: int, children: List[Node]):
        super().__init__(name, children)

    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:

        if self._name not in state or state[self._name]["result"] != RUNNING:
            child_position = 0
        else:
            child_position = state[self._name]["child"]

        while child_position < len(self._children):
            child = self._children[child_position]

            result = child.run(blackboard)
            if result == FAILED:
                return FAILED

            if result == RUNNING:
                state[self._name] = {
                    "result": RUNNING, "child": child_position}
                return RUNNING

            child_position = child_position + 1

        state[self._name] = {"result": SUCCEEDED}
        return SUCCEEDED


class Priority(Composite):
    def __init__(self, children: List[Node]):
        super().__init__(children)

    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        for child in self._children:
            result = child.run()

            if result == SUCCEEDED:
                state.clear()
                return SUCCEEDED

        # Please define the correct behavior of the composite
        return FAILED


class Selector(Composite):
    def __init__(self, children: List[Node]):
        super().__init__(children)

    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        # Please define the correct behavior of the composite
        for child in self._children:
            result = child.run(blackboard)
            if result == SUCCEEDED:
                return SUCCEEDED
        return FAILED

# You must define also the other composites here


# Task definitions


class FindHome(Task):
    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        print('FIND HOME Task')
        blackboard["HOME_PATH"] = "Left, Right, Straight, Left"

        return SUCCEEDED

# Same thing for the tasks


class GoHome(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('GO HOME Task')
        home_path = blackboard["HOME_PATH"]

        return SUCCEEDED


class Dock(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('DOCK Task')

        return SUCCEEDED


class CleanSpot(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('CLEAN SPOT Task')

        return SUCCEEDED


class DoneSpot(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('DONE SPOT Task')
        blackboard["SPOT_CLEANING"] = False

        return SUCCEEDED


class CleanFloor(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('CLEAN FLOOR Task')

        return SUCCEEDED


class DoneGeneral(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('DONE GENERAL Task')
        blackboard["GENERAL_CLEANING"] = False

        return SUCCEEDED


class DoNothing(Task):
    def run(self, blackboard: Dict[string, Any]) -> int:
        print('DO NOTHING Task')

        return SUCCEEDED
# Condition definitions

# Decorator definitions


class Timer(Decorator):
    _time: int

    def __init__(self, name: int, time: int, child: Node):
        super().__init__(name, child)
        self._time = time

    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        current_time = 0
        if self._name not in state or state[self._name] == -1:
            current_time = self._time
        else:
            current_time = state[self._name]

        current_time = current_time - 1
        if current_time == 0:
            state[self._name] = -1
            return SUCCEEDED

        result = self._child.run(state, blackboard)
        if result == FAILED:
            state[self._name] = -1
            return FAILED

        state[self._name] = current_time
        return RUNNING


class UntilFails(Decorator):
    _child: Node

    def __init__(self, child: Node):
        super().__init__(child)

    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        result = self._child.run(blackboard)
        if result == SUCCEEDED:
            state[self._name] = RUNNING
            return RUNNING
        return SUCCEEDED

# Define all the other conditions here as well


class BatterLessThan30(Condition):
    def run(self, state: Dict[string, Any], blackboard: Dict[string, Any]) -> int:
        return SUCCEEDED if blackboard["BATTERY_LEVEL"] < 30 else FAILED


class Spot(Condition):
    def run(self, blackboard: Dict[string, Any]) -> int:
        return SUCCEEDED if blackboard["SPOT_CLEANING"] == True else FAILED


class GeneralCleaning(Condition):
    def run(self, blackboard: Dict[string, Any]) -> int:
        return SUCCEEDED if blackboard["GENERAL_CLEANING"] == True else FAILED


class DustySpot(Condition):
    def run(self, blackboard: Dict[string, Any]) -> int:
        return SUCCEEDED if blackboard["DUSTY_SPOT"] == True else FAILED


# Now you have to instantiate the tree

tree = Priority([
    Sequence([
        BatterLessThan30(),
        FindHome(),
        GoHome(),
        Dock()
    ]),
    Selector([
        Sequence([
            Spot(),
            Timer(20, CleanSpot()),
            DoneSpot()
        ]),
        Sequence([
            GeneralCleaning(),
            Sequence([
                Priority([
                    Sequence([
                        DustySpot(),
                        Timer(35, CleanSpot())
                        ]),
                    UntilFails(CleanFloor())
                ]),
                DoneGeneral()
            ])
        ])
    ]),
    DoNothing()
])

# Main body of the assignment
current_blackboard: Dict[string, Any] = {
    "BATTERY_LEVEL": 20,
    "SPOT_CLEANING": False,
    "GENERAL_CLEANING": True,
    "DUSTY_SPOT": False,
    "HOME_PATH": ""
}
current_state: Dict[int, Any] = {
}
cycles = 10

while cycles > 0:
    # Changing the environment
    current_blackboard["BATTERY_LEVEL"] = current_blackboard["BATTERY_LEVEL"] - 3

    # Evaluating the tree
    result = tree.run(current_state, current_blackboard)

    # Going through the cycles
    cycles = cycles - 1
