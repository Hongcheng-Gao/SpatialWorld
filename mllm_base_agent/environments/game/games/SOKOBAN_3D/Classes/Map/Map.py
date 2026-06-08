# # Import Another Classes
from Classes.Map.Goal import *
from Classes.Map.Objective import *
from Classes.Map.Player import *
from random import randrange

class Map:
    def __init__(self, size):
        # Init Map Size, Init Player, Init Goals Count
        self.size = size
        self.player = Player(0,0)
        self.goals_count = 0
        self.objectives = Objective(0,0)
        self.goals = Goal(0,0)

        # Build Empty Array
        self.tiles = []
        for i in range(size):
            self.tiles.append([0 for j in range(size)])

        # Print Results
        print("Map Initialized With Size",size)
        # self.print_map()

    def print_map(self):
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[0])):
                print("[" + str(self.tiles[i][j]) + "]", end="");
            print("")

    def player_move(self, direction):
        if direction == "up" or direction == "UP":
            self.tiles[self.player.y][self.player.x] = 0
            self.player.move_up()
            self.player.steps += 1
            if self.player.y <= 0:
                self.player.move_down()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 1:
                self.player.move_down()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 3:
                self.objectives.move_up()
                if self.tiles[self.objectives.y][self.objectives.x] == 1:
                    self.objectives.move_down()
                    self.player.move_down()
                    self.player.steps -= 1
            self.tiles[self.goals.y][self.goals.x] = 4
            self.tiles[self.objectives.y][self.objectives.x] = 3
            self.tiles[self.player.y][self.player.x] = 2
        elif direction == "down" or direction == "DOWN":
            self.tiles[self.player.y][self.player.x] = 0
            self.player.move_down()
            self.player.steps += 1
            if self.player.y >= (len(self.tiles[0]) - 1):
                self.player.move_up()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 1:
                self.player.move_up()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 3:
                self.objectives.move_down()
                if self.tiles[self.objectives.y][self.objectives.x] == 1:
                    self.objectives.move_up()
                    self.player.move_up()
                    self.player.steps -= 1
            self.tiles[self.goals.y][self.goals.x] = 4
            self.tiles[self.objectives.y][self.objectives.x] = 3
            self.tiles[self.player.y][self.player.x] = 2
        elif direction == "left" or direction == "LEFT":
            self.tiles[self.player.y][self.player.x] = 0
            self.player.move_left()
            self.player.steps += 1
            if self.player.x <= 0:
                self.player.move_right()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 1:
                self.player.move_right()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 3:
                self.objectives.move_left()
                if self.tiles[self.objectives.y][self.objectives.x] == 1:
                    self.objectives.move_right()
                    self.player.move_right()
                    self.player.steps -= 1
            self.tiles[self.goals.y][self.goals.x] = 4
            self.tiles[self.objectives.y][self.objectives.x] = 3
            self.tiles[self.player.y][self.player.x] = 2
        elif direction == "right" or direction == "RIGHT":
            self.tiles[self.player.y][self.player.x] = 0
            self.player.move_right()
            self.player.steps += 1
            if self.player.x >= (len(self.tiles[0]) - 1):
                self.player.move_left()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 1:
                self.player.move_left()
                self.player.steps -= 1
            elif self.tiles[self.player.y][self.player.x] == 3:
                self.objectives.move_right()
                if self.tiles[self.objectives.y][self.objectives.x] == 1:
                    self.objectives.move_left()
                    self.player.move_left()
                    self.player.steps -= 1
            self.tiles[self.goals.y][self.goals.x] = 4
            self.tiles[self.objectives.y][self.objectives.x] = 3
            self.tiles[self.player.y][self.player.x] = 2

class Map_Builder:

    game_mode = 0
    current_level = 0

    def __init__(self):
        print("Map Builder is Initialized")

    def build_map(self):
        if self.mode == 0:
            # Build First Map
            print("")
            print("###################")
            print("BUILDING MAP {}".format(self.current_level))
            print("###################")
            return self.build_level_map()
        elif self.mode == 1:
            # Build Random Map
            print("")
            print("#######################")
            print("BUILDING RANDOM MAP {}".format(self.current_level))
            print("#######################")
            return self.build_random_map()

    def build_level_map(self):
        print("Building Map")
        
        if self.current_level == 0:
            # Level Pertama
            new_map = Map(15)
            new_map.tiles = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,1,0,0,1,1,1,1,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,1,1,1,1,0,0,0,0,0,0,0,1,0,1],
            [1,1,1,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,1,1,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,1,1,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,1,1,0,0,0,0,0,1,1,1,1,1,0,1],
            [1,1,1,0,0,0,0,0,1,1,1,1,1,0,1],
            [1,1,1,0,0,0,0,0,1,1,1,1,1,0,1],
            [1,1,1,0,0,0,0,0,1,1,1,1,1,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
            self.place_player_at(new_map,2,4)
            self.place_goal_at(new_map,13,13)
            self.place_objective_at(new_map,4,12)
        elif self.current_level == 1:
            new_map = Map(15)
            new_map.tiles =[
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,3,0,0,1,0,0,1,1,1,0,0,1],
            [1,0,0,1,1,0,1,0,0,1,1,1,0,0,1],
            [1,0,0,0,0,0,0,1,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,1,1,0,0,0,0,0,1],
            [1,0,0,1,1,1,1,1,1,0,0,0,0,0,1],
            [1,0,0,1,1,1,1,1,1,0,0,0,0,0,1],
            [1,0,0,1,1,0,0,0,0,0,0,1,1,1,1],
            [1,0,0,1,1,0,0,0,1,1,0,1,1,1,1],
            [1,0,0,0,1,1,0,0,1,1,0,0,0,0,1],
            [1,0,0,0,1,1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,4,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
            self.place_player_at(new_map,1,1)
            self.place_goal_at(new_map,13,13)
            self.place_objective_at(new_map,3,3)
        elif self.current_level == 2:
            new_map = Map(22)
            new_map.tiles = [
            [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,0,1,0,1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,1],
            [1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,1],
            [0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1],
            [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]]
            self.place_player_at(new_map,12,8)
            self.place_goal_at(new_map,20,8)
            self.place_objective_at(new_map,5,2)
        elif self.current_level == 3:
            new_map = Map(20)
            new_map.tiles = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1],
            [1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
            self.place_player_at(new_map,15,18)
            self.place_goal_at(new_map,10,9)
            self.place_objective_at(new_map,17,17)
        elif self.current_level == 4:
            new_map = Map(25)
            new_map.tiles = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,4,1,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,0,0,0,0],
            [1,0,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0],
            [1,0,1,1,1,1,0,1,1,1,1,0,2,0,0,1,0,0,0,0,0],
            [1,0,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
            [1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
            [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
            [0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1],
            [0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,1,0,1,0,1],
            [0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
            [0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,1,0,1,0,0,1],
            [0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,1],
            [0,0,0,1,0,1,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0],
            [0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,3,0,0,1,0],
            [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]]
            self.place_player_at(new_map,12,5)
            self.place_goal_at(new_map,14,1)
            self.place_objective_at(new_map,16,17)
        else:
            print("Story Mode Done!")
            # quit()
            return "DONE"

        return new_map

    def place_player_at(self, map, coord_x, coord_y):
        map.player = Player(coord_x, coord_y)
        map.tiles[map.player.y][map.player.x] = 2

    def place_goal_at(self, map, coord_x, coord_y):
        map.goals = Goal(coord_x,coord_y)
        map.tiles[map.goals.y][map.goals.x] = 4

    def place_objective_at(self, map, coord_x, coord_y):
        map.objectives = Objective(coord_x, coord_y)
        map.tiles[map.objectives.y][map.objectives.x] = 3

    def build_random_map(self):
        print("Building Random Map")
        random_size = randrange(7,15)
        new_map = Map(random_size)
        self.build_walls(new_map)
        self.place_random_player(new_map)
        self.place_random_goal(new_map)
        self.place_random_objective(new_map)
        return new_map

    def build_walls(self, map):
        tiles = map.tiles
        for i in range(len(tiles)):
            for j in range(len(tiles[0])):
                if i == 0 or j == 0 or i == len(tiles)-1 or j == len(tiles[0])-1:
                    tiles[i][j] = 1
        print("Build Walls Success")

    def place_random_player(self, map):
        self.place_player_at(map, randrange(1,len(map.tiles[0])-2),randrange(1,len(map.tiles)-2))

    def place_random_goal(self, map):
        rand_x = 0
        rand_y = 0

        while map.tiles[rand_y][rand_x] != 0:
            rand_x = randrange(1,len(map.tiles[0])-2)
            rand_y = randrange(1,len(map.tiles)-2)
        self.place_goal_at(map, rand_x,rand_y)

    def place_random_objective(self, map):
        rand_x = 0
        rand_y = 0
        while map.tiles[rand_y][rand_x] != 0:
            rand_x = randrange(2,len(map.tiles[0])-3)
            rand_y = randrange(2,len(map.tiles)-3)
        self.place_objective_at(map, rand_x, rand_y)

    def set_mode(self, mode):
        self.mode = mode
        print("Build Mode Set To : {}".format(self.mode))

