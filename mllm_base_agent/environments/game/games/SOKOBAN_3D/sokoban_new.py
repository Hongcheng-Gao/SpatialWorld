# IMPORT PYGAME AND OPENGL
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT PYGAME MENU
import pygame_menu
from pygame_menu.locals import *

# IMPORT CLASSES
import Classes.Map as Map
import Classes.Score as Score

# IMPORT GUI
import UI.MainMenu as MainMenuUI
import UI.NameInput as NameInputUI
import UI.Highscore as HighscoreUI
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

# IMPORT PICKLE
import pickle
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "Saves" / "savedata.bin"
FONT_FILE = SCRIPT_DIR / "Fonts" / "Roboto.ttf"

def main(mode, player_name):

    # INIT PYGAME
    pygame.init()
    display = (800, 600)
    clock = pygame.time.Clock()

    # INIT PYGAME DISPLAY AND OPENGL
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, display[0] / display[1], 0.1, 50.0)
    glTranslate(1,0,-5)
    glRotatef(45,1,0,0)
    glOrtho(0,800,0,800,0,1000)
    glEnable(GL_DEPTH_TEST)

    # INIT FONT
    pygame.font.init()
    
    # Debug Option
    enable_fps_counter = True

    # RENDER POSITION
    rotate_x = 0
    rotate_y = 0
    translate_x = 0
    translate_y = 0
    z_position = 0

    # MOUSE INPUTS
    mouse_rotate = False
    mouse_move = False

    # Set Game Mode
    # Mode 0 : Story
    # Mode 1 : Random
    game_mode = mode

    # CREATE CLASSES
    builder = Map.Map_Builder()
    builder.set_mode(game_mode)
    # builder.current_level = 4
    hasil_build = builder.build_map()
    steps_history = []
    # print(hasil_build)

    # Create Countdown Timer
    time_elapsed = 0
    clocktick = 0
    game_time = 30

    # Ger Global Ref
    global highscores

    # MAIN GAME LOOP
    pygame.key.set_repeat(16,100)
    in_game = True

    while in_game:
        clock.tick(120)
        clocktick += clock.get_rawtime()
        time_elapsed = clocktick // 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                in_game = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_rotate = True
                elif event.button == 3:
                    mouse_move = True
                elif event.button == 4: 
                    translate_y += 25
                elif event.button == 5: 
                    translate_y -= 25
            elif event.type == MOUSEBUTTONUP:
                mouse_rotate = False
                mouse_move = False
            elif event.type == MOUSEMOTION:
                i, j = event.rel
                if mouse_move:
                    translate_x += i
                    translate_y -= j
                elif mouse_rotate:
                    rotate_x += i
                    rotate_y += j
            elif event.type == KEYDOWN:
                # print(event)
                if event.key == 114:
                    hasil_build = builder.build_map()
                elif event.key == 119:
                    hasil_build.player_move("up")
                elif event.key == 97:
                    hasil_build.player_move("left")
                elif event.key == 115:
                    hasil_build.player_move("down")
                elif event.key == 100:
                    hasil_build.player_move("right")
                elif event.key == 273:
                    z_position -= 50
                elif event.key == 274:
                    z_position += 50
                elif event.key == 276:
                    translate_x -= 50
                elif event.key == 275:
                    translate_x += 50
                elif event.key == 113:
                    rotate_x -= 50
                elif event.key == 101:
                    rotate_x += 50

        # print("OBJECTIVE COORDINATE : ({0},{1})".format(hasil_build.objectives.x, hasil_build.objectives.y))
        # print("Player Steps : {}".format(hasil_build.player.steps))
        # CHECK GOAL
        if hasil_build.tiles[hasil_build.goals.y][hasil_build.goals.x] == 3:
            steps_history.append(hasil_build.player.steps)
            builder.current_level += 1
            hasil_build = builder.build_map()

        if time_elapsed >= game_time and game_mode == 1:
            hasil_build = "DONE"
        if hasil_build == "DONE":
            in_game = False
            if mode == 0:
                highscores.append(Score.StoryScore(player_name, steps_history))
                highscores = sort_highscore()
                save_save()
            elif mode == 1:
                highscores.append(Score.TimeAttackScore(player_name, builder.current_level))
                highscores = sort_highscore()
                save_save()

        if in_game:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_map(hasil_build)

            # TRANSLATE OBJECT IF MOUSE MOVED
            glTranslatef(translate_x, translate_y, -z_position)
            glRotatef(rotate_y/20.,1,0,0)
            glRotatef(rotate_x/20.,0,1,0)
            
            # RESET ROTATE
            rotate_x = 0
            rotate_y = 0
            translate_x = 0
            translate_y = 0
            z_position = 0

            if enable_fps_counter:
                draw_text("FPS: {}".format(clock.get_fps()), -display[0], display[1], -5, 12)

            if game_mode == 1: 
                draw_text("Time Remaining: {}".format((game_time-time_elapsed)), -display[0], display[1]+display[1]//4, -5, 64)
                draw_text("Score : {}".format(builder.current_level), -display[0], display[1]+350, -5, 16)
                # print("Time Elapsed  : {}\nTime Remaining: {}".format(time_elapsed, (game_time-time_elapsed)))
            pygame.display.flip()
        else:
            pygame.quit()
            MainWindow.show()
        
def draw_map(map):

    # Set Box Size
    box_size = 200 

    # Find Center Point Of The Map
    center_x = len(map.tiles) / 2 
    center_y = len(map.tiles[0])/ 2 
    # print("Center X: {}\nCenter Y: {}".format(center_x,center_y))

    # Set Cube Size
    Cube_Size = 150

    # Draw Plane
    for i in range(len(map.tiles)):
        for j in range(len(map.tiles[0])):
            draw_cube((j-center_x)*Cube_Size,-Cube_Size,(i-center_y)*-Cube_Size,Cube_Size,4)
            draw_plane((j-center_x)*Cube_Size,-Cube_Size,(i-center_y)*-Cube_Size,Cube_Size,0)

    # Draw Walls, Player, Objectives, and Goals
    # 1 : Walls
    # 2 : Player
    # 3 : Objectives
    # 4 : Goals
    for i in range(len(map.tiles)):
        for j in range(len(map.tiles[0])):
            if map.tiles[i][j] != 0 and map.tiles[i][j] != 4:
                draw_cube((j-center_x)*Cube_Size,0,(i-center_y)*-Cube_Size,Cube_Size,map.tiles[i][j])
                draw_plane((j-center_x)*Cube_Size,0,(i-center_y)*-Cube_Size,Cube_Size,map.tiles[i][j])
            elif map.tiles[i][j] == 4:
                draw_plane((j-center_x)*Cube_Size,0,(i-center_y)*-Cube_Size,Cube_Size,map.tiles[i][j])

def draw_cube( centerPosX, centerPosY, centerPosZ, edgeLength, mode):
    halfSideLength = edgeLength * 0.5
    vertices = (
        # front face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # back face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom left
        
        # left face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # right face
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # top face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # top face
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength  # bottom left
    )
    
    # OLD RENDER (POLYGONS)
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    
    # NEW RENDER (GL_QUADS)
    glBegin(GL_QUADS)
    if mode == 0:
        glColor3f(0,0,1)
    elif mode == 1:
        glColor3f(0,1,0)
    elif mode == 2:
        glColor3f(1,0,0)
    elif mode == 3:
        glColor3f(1,0,1)
    elif mode == 4:
        glColor3f(0.5,0.5,0.5)

    # FOR EVERY VERTICES
    for x in range(24):
        glVertex3f(vertices[x*3],vertices[x*3 + 1],vertices[x * 3 + 2])
    glColor3f(0,0,0)
    glEnd()

def draw_plane( centerPosX, centerPosY, centerPosZ, edgeLength, mode):
    halfSideLength = edgeLength * 0.5
    vertices = (
        # front face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # back face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom left
        
        # left face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # right face
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # top face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # bottom left
        
        # top face
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength  # bottom left
    )
    
    # OLD RENDER (POLYGONS)
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    glLineWidth(2)
    if mode == 1:
        glColor3f(0,0,0)
    elif mode == 0:
        glColor3f(1,1,1)
    elif mode == 4:
        glLineWidth(5)
        glColor3f(0,1,1)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3,GL_FLOAT,0,vertices)
    glDrawArrays(GL_QUADS,0,24)
    glDisableClientState(GL_VERTEX_ARRAY)

def play_normal():
    player_name = InputNameDialog.lineEdit.text()
    # print(player_name)
    Dialog.close()
    if player_name != '':
        MainWindow.hide()
        main(0, player_name)
    else:
        set_connect_play_normal()

def play_time_attack():
    player_name = InputNameDialog.lineEdit.text()
    # print(player_name)
    Dialog.close()
    if player_name != '':
        MainWindow.hide()
        main(1, player_name)
    else:
        set_connect_play_time_attack()

def show_highscore():
    HighscoreMainWindow.show()
    model = QtGui.QStandardItemModel(HighscoreWindow.listView)
    model2 = QtGui.QStandardItemModel(HighscoreWindow.listView_2)
    for score in highscores:
        item = QtGui.QStandardItem(score.get_data())
        if isinstance(score, Score.StoryScore):
            model.appendRow(item)
        elif isinstance(score, Score.TimeAttackScore):
            model2.appendRow(item)
    HighscoreWindow.listView.setModel(model)
    HighscoreWindow.listView_2.setModel(model2)

def close_highscore():
    HighscoreMainWindow.close()

def set_connect_play_normal():
    InputNameDialog.setupUi(Dialog)
    InputNameDialog.buttonBox.accepted.connect(play_normal)
    Dialog.show()

def set_connect_play_time_attack():
    InputNameDialog.setupUi(Dialog)
    InputNameDialog.buttonBox.accepted.connect(play_time_attack)
    Dialog.show()

def load_save():
    p = open(DATA_FILE, "rb")
    load_files = pickle.load(p)
    p.close()
    print("Success Load Data!")
    return load_files

def save_save():
    p = open(DATA_FILE, "wb")
    pickle.dump(highscores, p)
    p.close()
    print("Success Save Data!")

def quit_app():
    msgbox = QtWidgets.QMessageBox()
    msgbox.setIcon(QtWidgets.QMessageBox.Question)
    msgbox.setText("Are You Sure Want To Quit?")
    msgbox.setWindowTitle("Quit Confirmation")
    msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    if msgbox.exec_() == QtWidgets.QMessageBox.Yes:
        quit()
    else:
        msgbox.close()

def sort_highscore():
    story = []
    time_atk = []

    for score in highscores:
        if isinstance(score, Score.StoryScore):
            story.append(score)
        elif isinstance(score, Score.TimeAttackScore):
            time_atk.append(score)

    sorted_highscore = []
    story.sort(key=lambda x: x.get_score_ttl(), reverse=False)
    time_atk.sort(key=lambda x: x.levels, reverse=True)
    return story + time_atk

def draw_text(text, x_pos, y_pos, z_pos, size):
    font = pygame.font.Font(str(FONT_FILE), size)
    textSurface = font.render(text, True, (255,255,255,255), (0,0,0,255))     
    textData = pygame.image.tostring(textSurface, "RGBA", True)     
    glRasterPos3d(x_pos, y_pos, z_pos)     
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

# Create Global Var Highscore
highscores = []

# RESET HIGHSCORE
# save_save()

highscores = load_save()
highscores = sort_highscore() 

# CALL GUI
app = QtWidgets.QApplication(sys.argv)

# MainWindow
MainWindow = QtWidgets.QMainWindow()
MainMenuWindow = MainMenuUI.Ui_MainWindow()
MainMenuWindow.setupUi(MainWindow)

# Connect MainWindows Button To An Event Handler
MainMenuWindow.pushButton.clicked.connect(set_connect_play_normal)
MainMenuWindow.pushButton_2.clicked.connect(set_connect_play_time_attack)
MainMenuWindow.pushButton_3.clicked.connect(show_highscore)
MainMenuWindow.pushButton_4.clicked.connect(quit_app)

# SHOW MainWindow
MainWindow.show()

# Dialog
Dialog = QtWidgets.QDialog()
InputNameDialog = NameInputUI.Ui_Dialog()
InputNameDialog.setupUi(Dialog)

# Highscore Window
HighscoreMainWindow = QtWidgets.QMainWindow()
HighscoreWindow = HighscoreUI.Ui_HighscoreWindow()
HighscoreWindow.setupUi(HighscoreMainWindow)
HighscoreWindow.BtnBack.clicked.connect(close_highscore)

sys.exit(app.exec_())

# CALL MAIN
# main(1)
