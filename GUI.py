import PySimpleGUI as sg
import cv2

video_path = "video/Example2.mp4"
cap = cv2.VideoCapture(video_path)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
screen_width, screen_height = sg.Window.get_screen_size()
window_width = screen_width
window_height = screen_height
bc = '#323232' # background color

stats_layout = [
                [sg.Text('Analytics:', key='-title-', size=(20, 2), font='Helvetica 25', text_color='white',background_color=bc)],
                [sg.Text('Players on the screen:', key='-players-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-playersi-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Team1:', key='-players_team1-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-players_team1i-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Team2:', key='-players_team2-', size=(20, 2), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-players_team2i-', size=(5, 2), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Ball Possession:', key='-ball possession-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Ball possession Team 1:', key='-ball_possession1-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-ball_possession1i-', size=(8, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Ball possession Team 2:', key='-ball_possession2-', size=(20, 2), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-ball_possession2i-', size=(8, 2), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Tackles:', key='-tackles-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-tackles_ti-', size=(8, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Tackles team 1:', key='-tackles_team1-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-tackles_team1i-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Tackles team 2:', key='-tackles_team2-', size=(20, 2), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-tackles_team2i-', size=(5, 2), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('OUTs:', key='-outs-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('OUT team 1:', key='-outs_team1-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-outs_team1i-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('OUT team 2:', key='-outs_team2-', size=(20, 2), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-outs_team2i-', size=(5, 2), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Passages:', key='-passages-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-outs_team2i-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Passages team 1:', key='-passages_team1-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-passages_team1i-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)],
                [sg.Text('Passages team 2:', key='-passages_team2-', size=(20, 1), font='Helvetica 20', text_color='white',background_color=bc),sg.Text(key='-passages_team2i-', size=(5, 1), font='Helvetica 20', text_color='white',background_color=bc)]
            ]
image_layout=[         
    [sg.Image(size= (0.75*window_width,None), pad=(20, 20), background_color=bc, key='-image-')],   
]
layout = [[
     sg.Column(stats_layout, element_justification='left', size = (int(0.23*window_width),window_height),background_color= bc,vertical_alignment='center', pad=((20, 0), (int(video_height*0.06), 0))),
     sg.Column(image_layout, element_justification='center',background_color= bc,pad=((0, 20), (int(video_height*0.05), int(video_height*0.05))))
]]
